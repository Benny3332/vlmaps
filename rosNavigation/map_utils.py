import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import pyvisgraph as vg
import h5py
from pathlib import Path

from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple, Set, Union

def get_bbox(center, size):
    """
    Return min corner and max corner coordinate
    """
    min_corner = center - size / 2
    max_corner = center + size / 2
    return min_corner, max_corner

def get_dist_to_bbox_2d(center, size, pos):
    min_corner_2d, max_corner_2d = get_bbox(center, size)

    dx = pos[0] - center[0]
    dy = pos[1] - center[1]

    if pos[0] < min_corner_2d[0] or pos[0] > max_corner_2d[0]:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
            *  |  |  *
            ___|__|___
               |  |
            ___|__|___
               |  |
            *  |  |  *
            """

            dx_c = np.abs(dx) - size[0] / 2
            dy_c = np.abs(dy) - size[1] / 2
            dist = np.sqrt(dx_c * dx_c + dy_c * dy_c)
            return dist
        else:
            """
            star region
               |  |
            ___|__|___
            *  |  |  *
            ___|__|___
               |  |
               |  |
            """
            dx_b = np.abs(dx) - size[0] / 2
            return dx_b
    else:
        if pos[1] < min_corner_2d[1] or pos[1] > max_corner_2d[1]:
            """
            star region
               |* |
            ___|__|___
               |  |
            ___|__|___
               |* |
               |  |
            """
            dy_b = np.abs(dy) - size[1] / 2
            return dy_b

        """
        star region
           |  |  
        ___|__|___
           |* |   
        ___|__|___
           |  |   
           |  |  
        """
        return 0

def cvt_pose_vec2tf(pos_quat_vec: np.ndarray) -> np.ndarray:
    """
    pos_quat_vec: (px, py, pz, qx, qy, qz, qw)
    """
    pose_tf = np.eye(4)
    pose_tf[:3, 3] = pos_quat_vec[:3].flatten()
    rot = R.from_quat(pos_quat_vec[3:].flatten())
    pose_tf[:3, :3] = rot.as_matrix()
    # rot_change = R.from_euler('xyz', [-90, 0, -90], degrees=True)
    # rot_change = R.from_euler('xyz', [0, 90, 90], degrees=True)
    # rot = R.from_quat(pos_quat_vec[3:].flatten())
    # rot = rot * rot_change
    # pose_tf[:3, :3] = rot.as_matrix()
    return pose_tf

def base_pos2grid_id_3d(gs, cs, x_base, y_base, z_base):
    row = int(gs / 2 - int(x_base / cs))
    col = int(gs / 2 - int(y_base / cs))
    h = int(z_base / cs)
    return [row, col, h]

def grid_id2base_pos_3d(row, col, height, cs, gs):
    base_x = (gs / 2 - row) * cs
    base_y = (gs / 2 - col) * cs
    base_z = height * cs
    return [base_x, base_y, base_z]

def grid_id2base_pos_3d_batch(pos_grid_np, cs, gs):
    """
    pos_grid_np: [N, 3] np.int32
    """
    base_x = (gs / 2 - pos_grid_np[:, 0]) * cs
    base_y = (gs / 2 - pos_grid_np[:, 1]) * cs
    base_z = pos_grid_np[:, 2] * cs
    return [base_x, base_y, base_z]

def base_rot_mat2theta(rot_mat: np.ndarray) -> float:
    """Convert base rotation matrix to rotation angle (rad) assuming x is forward, y is left, z is up

    Args:
        rot_mat (np.ndarray): (3,3) rotation matrix

    Returns:
        float: rotation angle
    """
    theta = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    return theta

def find_similar_category_id(class_name, classes_list):
    if class_name in classes_list:
        return classes_list.index(class_name)
    import openai

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    classes_list_str = ",".join(classes_list)
    client = openai.OpenAI(api_key=openai_key,base_url='https://api.gptsapi.net/v1')
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": "You are a helpful assistant that can answer questions about objects and their categories.You can only answer one most relevant word."
            },
            {
                "role": "assistant",
                "content": "OK."
            },
            {
                "role": "user",
                "content": "What is television most relevant to among tv_monitor,plant,chair",
            },
            {
                "role": "assistant",
                "content": "tv_monitor"
            },
            {
                "role": "user",
                "content": "What is drawer most relevant to among tv_monitor,chest_of_drawers,chair"
            },
            {
                "role": "assistant",
                "content": "chest_of_drawer"
            },
            {
                "role": "user",
                "content": f"What is {class_name} most relevant to among {classes_list_str}"
            }
        ],
        max_tokens=300,
    )

    text = response.choices[0].message.content
    print(text)
    return classes_list.index(text)

def build_visgraph_with_obs_map(obs_map, use_internal_contour=False, internal_point=None, vis=False):
    # 将障碍物地图转换为可视化图像
    obs_map_vis = (obs_map[:, :, None] * 255).astype(np.uint8)
    obs_map_vis = np.tile(obs_map_vis, [1, 1, 3])
    # 如果开启可视化，则显示障碍物地图
    if vis:
        cv2.imshow("obs", obs_map_vis)
        cv2.waitKey()

    # 获取障碍物地图中的轮廓、中心点、边界框和层次结构
    contours_list, centers_list, bbox_list, hierarchy = get_segment_islands_pos(
        obs_map, 0, detect_internal_contours=use_internal_contour
    )

    poly_list = []

    # 遍历所有轮廓
    for contour in contours_list:
        # 如果开启可视化，则绘制轮廓
        if vis:
            contour_cv2 = contour[:, [1, 0]]
            cv2.drawContours(obs_map_vis, [contour_cv2], 0, (0, 255, 0), 3)
            cv2.imshow("obs", obs_map_vis)
        # 提取轮廓点
        contour_pos = []
        for [row, col] in contour:
            contour_pos.append(vg.Point(row, col))
        # 将轮廓点添加到多边形列表中
        poly_list.append(contour_pos)
        # 提取轮廓点的x和z坐标
        xlist = [x.x for x in contour_pos]
        zlist = [x.y for x in contour_pos]
        # 如果开启可视化，则绘制轮廓点的x和z坐标曲线
        if vis:
            # plt.plot(xlist, zlist)

            cv2.waitKey()
    # 创建可视化图
    g = vg.VisGraph()
    # 构建可视化图
    g.build(poly_list, workers=4)
    # 返回可视化图
    return g

def get_segment_islands_pos(segment_map, label_id, detect_internal_contours=False):
    mask = segment_map == label_id
    mask = mask.astype(np.uint8)
    detect_type = cv2.RETR_EXTERNAL
    if detect_internal_contours:
        detect_type = cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(mask, detect_type, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours back to numpy index order
    contours_list = []
    for contour in contours:
        tmp = contour.reshape((-1, 2))
        tmp_1 = np.stack([tmp[:, 1], tmp[:, 0]], axis=1)
        contours_list.append(tmp_1)

    centers_list = []
    bbox_list = []
    for c in contours_list:
        xmin = np.min(c[:, 0])
        xmax = np.max(c[:, 0])
        ymin = np.min(c[:, 1])
        ymax = np.max(c[:, 1])
        bbox_list.append([xmin, xmax, ymin, ymax])

        centers_list.append([(xmin + xmax) / 2, (ymin + ymax) / 2])

    return contours_list, centers_list, bbox_list, hierarchy

def point_in_contours(obs_map, contours_list, point):
    """
    obs_map: np.ndarray, 1 free, 0 occupied
    contours_list: a list of cv2 contours [[(col1, row1), (col2, row2), ...], ...]
    point: (row, col)
    """
    row, col = int(point[0]), int(point[1])
    ids = []
    print("contours num: ", len(contours_list))
    for con_i, contour in enumerate(contours_list):
        contour_cv2 = contour[:, [1, 0]]
        con_mask = np.zeros_like(obs_map, dtype=np.uint8)
        cv2.drawContours(con_mask, [contour_cv2], 0, 255, -1)
        # con_mask_copy = con_mask.copy()
        # cv2.circle(con_mask_copy, (col, row), 10, 0, 3)
        # cv2.imshow("contour_mask", con_mask_copy)
        # cv2.waitKey()
        if con_mask[row, col] == 255:
            ids.append(con_i)

    return ids

def find_closest_points_between_two_contours(obs_map, contour_a, contour_b):
    a = np.zeros_like(obs_map, dtype=np.uint8)
    b = np.zeros_like(obs_map, dtype=np.uint8)
    cv2.drawContours(a, [contour_a[:, [1, 0]]], 0, 255, 1)
    cv2.drawContours(b, [contour_b[:, [1, 0]]], 0, 255, 1)
    rows_a, cols_a = np.where(a == 255)
    rows_b, cols_b = np.where(b == 255)
    pts_a = np.concatenate([rows_a.reshape((-1, 1)), cols_a.reshape((-1, 1))], axis=1)
    pts_b = np.concatenate([rows_b.reshape((-1, 1)), cols_b.reshape((-1, 1))], axis=1)
    dists = cdist(pts_a, pts_b)
    id = np.argmin(dists)
    ida, idb = np.unravel_index(id, dists.shape)
    return [rows_a[ida], cols_a[ida]], [rows_b[idb], cols_b[idb]]

def load_3d_map(map_path: str) -> Tuple[Set[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 3D voxel map with features

    Args:
        map_path (str): path to save the map as an H5DF file.
    Return:
        mapped_iter_list (Set[int]): stores already processed frame's number.
        grid_feat (np.ndarray): (N, feat_dim) features of each 3D point.
        grid_pos (np.ndarray): (N, 3) each row is the (row, col, height) of an occupied cell.
        weight (np.ndarray): (N,) accumulated weight of the cell's features.
        occupied_ids (np.ndarray): (gs, gs, vh) either -1 or 1. 1 indicates
            occupation.
        grid_rgb (np.ndarray, optional): (N, 3) each row stores the rgb value
            of the cell.
        ---
        N is the total number of occupied cells in the 3D voxel map.
        gs is the grid size (number of cells on each side).
        vh is the number of cells in height direction.
    """
    with h5py.File(map_path, "r") as f:
        mapped_iter_list = f["mapped_iter_list"][:].tolist()
        grid_feat = f["grid_feat"][:]
        grid_pos = f["grid_pos"][:]
        weight = f["weight"][:]
        occupied_ids = f["occupied_ids"][:]
        grid_rgb = None
        if "grid_rgb" in f:
            grid_rgb = f["grid_rgb"][:]
        return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb

def cam_load_3d_map(map_path: Union[Path, str]):
    with h5py.File(map_path, "r") as f:
        mapped_iter_list = f["mapped_iter_list"][:].tolist()
        grid_feat = f["grid_feat"][:]
        grid_pos = f["grid_pos"][:]
        weight = f["weight"][:]
        occupied_ids = f["occupied_ids"][:]
        grid_rgb = f["grid_rgb"][:]
        pcd_min = f["pcd_min"][:]
        pcd_max = f["pcd_max"][:]
        cs = f["cs"][()]
        return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb, pcd_min, pcd_max, cs

def pool_3d_label_to_2d(mask_3d: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    mask_2d = np.zeros((gs, gs), dtype=bool)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask_2d[row, col] = mask_3d[i] or mask_2d[row, col]

    return mask_2d

def get_segment_islands_pos(segment_map, label_id, detect_internal_contours=False):
    mask = segment_map == label_id
    mask = mask.astype(np.uint8)
    detect_type = cv2.RETR_EXTERNAL
    if detect_internal_contours:
        detect_type = cv2.RETR_TREE

    contours, hierarchy = cv2.findContours(mask, detect_type, cv2.CHAIN_APPROX_SIMPLE)
    # convert contours back to numpy index order
    contours_list = []
    for contour in contours:
        tmp = contour.reshape((-1, 2))
        tmp_1 = np.stack([tmp[:, 1], tmp[:, 0]], axis=1)
        contours_list.append(tmp_1)

    centers_list = []
    bbox_list = []
    for c in contours_list:
        xmin = np.min(c[:, 0])
        xmax = np.max(c[:, 0])
        ymin = np.min(c[:, 1])
        ymax = np.max(c[:, 1])
        bbox_list.append([xmin, xmax, ymin, ymax])

        centers_list.append([(xmin + xmax) / 2, (ymin + ymax) / 2])

    return contours_list, centers_list, bbox_list, hierarchy