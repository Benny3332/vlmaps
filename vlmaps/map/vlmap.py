from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import gdown
import logging
from tqdm import tqdm
import clip
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import torch
from vlmaps.utils.clip_utils import get_text_feats_multiple_templates
from vlmaps.utils.visualize_utils import pool_3d_label_to_2d

# from utils.ai2thor_constant import ai2thor_class_list
# from utils.clip_mapping_utils import load_map
# from utils.planning_utils import (
#     find_similar_category_id,
#     get_dynamic_obstacles_map,
#     get_lseg_score,
#     get_segment_islands_pos,
#     mp3dcat,
#     segment_lseg_map,
# )
from vlmaps.map.vlmap_builder import VLMapBuilder
from vlmaps.map.vlmap_builder_2 import VLMapBuilder2
from vlmaps.map.vlmap_builder_cam import VLMapBuilderCam
from vlmaps.utils.mapping_utils import load_3d_map
from vlmaps.map.map import Map
from vlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from vlmaps.utils.clip_utils import get_lseg_score


class VLMap(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        super().__init__(map_config, data_dir=data_dir)
        self.scores_mat = None
        self.categories = None

    def create_map(self, data_dir: Union[Path, str]) -> None:
        print(f"Creating map for scene at: ", data_dir)
        self._setup_paths(data_dir)
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_builder = VLMapBuilder(
                self.data_dir,
                self.map_config,
                self.pose_path,
                self.rgb_paths,
                self.depth_paths,
                self.base2cam_tf,
                self.base_transform,
            )
            self.map_builder.create_mobile_base_map()
        if self.map_config.pose_info.pose_type == "mobile_base_2":
            self.map_builder = VLMapBuilder2(
                self.data_dir,
                self.map_config,
                self.pose_path,
                self.rgb_paths,
                self.depth_paths,
                self.base2cam_tf,
                self.base_transform,
            )
            self.map_builder.create_mobile_base_map()
        elif self.map_config.pose_info.pose_type == "camera_base":
            self.map_builder = VLMapBuilderCam(
                self.data_dir,
                self.map_config,
                self.pose_path,
                self.rgb_paths,
                self.depth_paths,
                self.base2cam_tf,
                self.base_transform,
            )
            self.map_builder.create_camera_map()
        else:
            raise ValueError("Invalid pose type")

    def load_map(self, data_dir: str) -> bool:
        self._setup_paths(data_dir)
        print(self.data_dir)
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_save_path = Path(data_dir) / "vlmap" / "vlmaps.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
            ) = load_3d_map(self.map_save_path)
        if self.map_config.pose_info.pose_type == "mobile_base_2":
            self.map_save_path = Path(data_dir) / "vlmap" / "vlmaps.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
            ) = load_3d_map(self.map_save_path)
        elif self.map_config.pose_info.pose_type == "camera_base":
            self.map_save_path = Path(data_dir) / "vlmap_cam" / "vlmaps_cam.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
                self.pcd_min,
                self.pcd_max,
                self.cs,
            ) = VLMapBuilderCam.load_3d_map(self.map_save_path)
        else:
            raise ValueError("Invalid pose type")

        return True

    def _init_clip(self, clip_version="ViT-B/32"):
        if hasattr(self, "clip_model"):
            print("clip model is already initialized")
            return
        if torch.cuda.is_available():
            self.device = "cuda:1"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[self.clip_version]
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(self.clip_version)  # clip.available_models()
        self.clip_model.to(self.device).eval()

    def init_categories(self, categories: List[str]) -> np.ndarray:
        self.categories = categories
        self.scores_mat = get_lseg_score(
            self.clip_model,
            self.categories,
            self.grid_feat,
            self.clip_feat_dim,
            use_multiple_templates=True,
            add_other=True,
        )  # score for name and other
        vlmaps_data_dir = self.data_dir
        # save_path = vlmaps_data_dir / "vlmap_cam" / "scores_mat.npy"
        # np.save(save_path, self.scores_mat)
        # print(f"{save_path} is saved.")
        return self.scores_mat

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            cat_id = 0
        # logging.info(f"self.categories: {self.categories}")
        # logging.info(f"cat_id: {cat_id}")
        # logging.info(f"catscores_mat_id: {scores_mat.shape}")
        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask

    def customize_obstacle_map(
        self,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        vis: bool = False,
    ):
        if self.obstacles_cropped is None and self.obstacles_map is None:
            self.generate_obstacle_map()
        if not hasattr(self, "clip_model"):
            print("init_clip in customize obstacle map")
            self._init_clip()

        self.obstacles_new_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            self.obstacles_cropped,
            self.map_config.potential_obstacle_names,
            self.map_config.obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.clip_feat_dim,
            vis=vis,
        )
        # 对一个二值地图（binary_map）进行膨胀处理，同时可选地应用高斯滤波
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
        )
        # 所有原来等于0的元素对应的位置会是True，而不等于0的元素对应的位置会是False
        self.obstacles_new_cropped = self.obstacles_new_cropped == 0

    # def load_categories(self, categories: List[str] = None):
    #     if categories is None:
    #         if self.map_config["categories"] == "mp3d":
    #             categories = mp3dcat.copy()
    #         elif self.map_config["categories"] == "ai2thor":
    #             categories = ai2thor_class_list.copy()

    #     predicts = segment_lseg_map(self.clip_model, categories, self.map_cropped, self.clip_feat_dim)
    #     no_map_mask = self.obstacles_new_cropped > 0  # free space in the map

    #     self.labeled_map_cropped = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))
    #     self.labeled_map_cropped[no_map_mask] = -1
    #     labeled_map = -1 * np.ones((self.map.shape[0], self.map.shape[1]))

    #     labeled_map[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1] = self.labeled_map_cropped

    #     self.categories = categories
    #     self.labeled_map_full = labeled_map

    # def load_region_categories(self, categories: List[str]):
    #     if "other" not in categories:
    #         self.region_categories = ["other"] + categories
    #     predicts = segment_lseg_map(
    #         self.clip_model, self.region_categories, self.map_cropped, self.clip_feat_dim, add_other=False
    #     )
    #     self.labeled_region_map_cropped = predicts.reshape((self.xmax - self.xmin + 1, self.ymax - self.ymin + 1))

    # def get_region_predict_mask(self, name: str) -> np.ndarray:
    #     assert self.region_categories
    #     cat_id = find_similar_category_id(name, self.region_categories)
    #     mask = self.labeled_map_cropped == cat_id
    #     return mask

    # def get_predict_mask(self, name: str) -> np.ndarray:
    #     cat_id = find_similar_category_id(name, self.categories)
    #     return self.labeled_map_cropped == cat_id

    # def get_distribution_map(self, name: str) -> np.ndarray:
    #     assert self.categories
    #     cat_id = find_similar_category_id(name, self.categories)
    #     if self.scores_map is None:
    #         scores_list = get_lseg_score(self.clip_model, self.categories, self.map_cropped, self.clip_feat_dim)
    #         h, w = self.map_cropped.shape[:2]
    #         self.scores_map = scores_list.reshape((h, w, len(self.categories)))
    #     # labeled_map_cropped = self.labeled_map_cropped.copy()
    #     return self.scores_map[:, :, cat_id]

    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        assert self.categories
        # cat_id = find_similar_category_id(name, self.categories)
        # labeled_map_cropped = self.scores_mat.copy()  # (N, C) N: number of voxels, C: number of categories
        # labeled_map_cropped = np.argmax(labeled_map_cropped, axis=1)  # (N,)
        # pc_mask = labeled_map_cropped == cat_id # (N,)
        # self.grid_pos[pc_mask]
        pc_mask = self.index_map(name, with_init_cat=True)
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        # print(f"showing mask for object cat {name}")
        # cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        # cv2.imshow(f"mask_{name}_gaussian", (foreground * 255).astype(np.uint8))
        foreground = binary_dilation(foreground)
        # cv2.imshow(f"mask_{name}_processed", (foreground.astype(np.float32) * 255).astype(np.uint8))
        # cv2.waitKey()

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)
        # print("centers", centers)

        # whole map position
        for i in range(len(contours)):
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        return contours, centers, bbox_list
    
    def get_pos_and_color(self, name: str, vis: bool = False) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], List[Dict]]:
        """
        Get the contours, centers, bbox list and color distributions of a certain category
        on a full map
        """
        assert self.categories
        # 获取目标类别的3D点云掩码
        pc_mask = self.index_map(name, with_init_cat=True)
        # pc_mask_index = np.where(pc_mask)[0]
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]
        # mask_2d_index = np.stack(np.where(mask_2d), axis=1)
        if vis:
            cv2.imshow(f"mask_{name}", (mask_2d.astype(np.float32) * 255).astype(np.uint8))
            cv2.waitKey()
        # 创建彩色mask图像（裁剪区域大小）
        color_mask = np.zeros((mask_2d.shape[1], mask_2d.shape[0], 3), dtype=np.uint8)
        
        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        foreground = binary_dilation(foreground)
        # foreground_index = np.stack(np.where(foreground), axis=1)
        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)

        contours_reverse = [None] * len(contours)
        for i in range(len(contours)):
            contours_reverse[i] = contours[i][:, [1, 0]]  # 从 (row,col) 转为 (col,row)
        
        # 存储每个物体的颜色分布信息
        color_distributions = []
        
        # 为每个轮廓创建点云索引列表
        contour_indices = [[] for _ in range(len(contours))]
        r_c = []
        local_r_c = []
        # 遍历所有属于目标类别的体素
        for idx in np.where(pc_mask)[0]:
            # 获取体素的全局坐标
            r, c, h = self.grid_pos[idx]
            r_c.append([r,c])
            # 转换为裁剪区域坐标
            local_r = r - self.rmin
            local_c = c - self.cmin
            local_r_c.append([local_r,local_c])
            # 检查是否在裁剪区域内
            if 0 <= local_r < foreground.shape[0] and 0 <= local_c < foreground.shape[1]:
                # 检查点是否在某个轮廓内
                point = (local_c, local_r)  # OpenCV格式 (x,y)
                
                for contour_idx, contour in enumerate(contours_reverse):
                    # 检查点是否在当前轮廓内
                    if cv2.pointPolygonTest(contour, point, False) >= 0:
                        contour_indices[contour_idx].append(idx)
                        break
        
        # 处理每个轮廓的点云
        for i, indices in enumerate(contour_indices):
            # 获取当前轮廓的点云索引
            obj_indices = indices
            
            # 计算颜色分布
            color_dist = {}
            if obj_indices:

                # mask = np.zeros(len(self.grid_pos), dtype=bool)
                # mask[obj_indices] = True
                # from vlmaps.utils.visualize_utils import visualize_masked_map_3d
                # visualize_masked_map_3d(self.grid_pos, mask, self.grid_rgb)

                obj_colors = self.grid_rgb[obj_indices]
                n_points = len(obj_colors)
                
                # 根据点数选择合适的聚类方法
                if n_points < 3:
                    # 点数太少，直接使用平均颜色
                    avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                    color_dist = {"main_colors": [{"color": avg_color, "proportion": 1.0}]}
                    main_color = tuple(int(c) for c in avg_color)
                else:
                    # 使用K-means聚类识别主要颜色
                    from sklearn.cluster import KMeans
                    try:
                        n_clusters = min(2, n_points)
                        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(obj_colors)
                        cluster_centers = kmeans.cluster_centers_.astype(int)
                        cluster_labels, counts = np.unique(kmeans.labels_, return_counts=True)
                        total = len(obj_colors)
                        
                        # 按占比排序颜色
                        sorted_indices = np.argsort(counts)[::-1]
                        main_colors = []
                        
                        for idx in sorted_indices:
                            color = cluster_centers[idx].tolist()
                            proportion = counts[idx] / total
                            main_colors.append({"color": color, "proportion": proportion})
                        
                        color_dist = {"main_colors": main_colors}
                        main_color = tuple(int(c) for c in main_colors[0]["color"])
                    except Exception as e:
                        print(f"KMeans聚类失败: {e}")
                        avg_color = np.mean(obj_colors, axis=0).astype(int).tolist()
                        color_dist = {"main_colors": [{"color": avg_color, "proportion": 1.0}]}
                        main_color = tuple(int(c) for c in avg_color)
            else:
                # 没有颜色数据，使用黑色
                color_dist = {"main_colors": [{"color": [0, 0, 0], "proportion": 1.0}]}
                main_color = (0, 0, 0)
            
            color_distributions.append(color_dist)
            
            # 使用主色填充物体区域（裁剪区域内坐标）
            bgr_color = (main_color[2], main_color[1], main_color[0])
            cv2.drawContours(color_mask, [contours[i].astype(np.int32)], -1, bgr_color, thickness=cv2.FILLED)
            
            # 转换坐标到全局地图（与原始代码保持一致）
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        # 可视化彩色mask（裁剪区域内）
        if vis:
            # 创建用于可视化的轮廓（转换回(row,col)格式以匹配mask_2d坐标系）
            visualization_contours = []
            for contour in contours:
                # 将轮廓从OpenCV格式(col,row)转回图像数组格式(row,col)
                vis_contour = contour[:, [1, 0]].copy()  # 交换回(row,col)
                visualization_contours.append(vis_contour.astype(np.int32))
            
            # 添加轮廓边界（白色）
            contour_mask = np.zeros_like(color_mask)
            for contour in visualization_contours:
                cv2.drawContours(contour_mask, [contour], -1, (255, 255, 255), 1)
            
            # 叠加轮廓到彩色mask
            combined_mask = cv2.addWeighted(color_mask, 0.8, contour_mask, 0.2, 0)
            combined_mask_display = combined_mask.transpose(1, 0, 2)
            # 显示彩色mask
            logging.debug(f"color_mask_name: {name}")
            cv2.imshow(f"color_mask", combined_mask_display)
            cv2.waitKey() 

        return contours, centers, bbox_list, color_distributions
