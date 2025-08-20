import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import colorcet as cc

def visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray):
    grid_rgb = rgb / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    o3d.visualization.draw_geometries([pcd])


def get_heatmap_from_mask_3d(
    pc: np.ndarray, mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01
) -> np.ndarray:
    target_pc = pc[mask, :]
    other_ids = np.where(mask == 0)[0]
    other_pc = pc[other_ids, :]

    target_sim = np.ones((target_pc.shape[0], 1))
    other_sim = np.zeros((other_pc.shape[0], 1))
    pbar = tqdm(other_pc, desc="Computing heat", total=other_pc.shape[0])
    for other_p_i, p in enumerate(pbar):
        dist = np.linalg.norm(target_pc - p, axis=1) / cell_size
        min_dist_i = np.argmin(dist)
        min_dist = dist[min_dist_i]
        other_sim[other_p_i] = np.clip(1 - min_dist * decay_rate, 0, 1)

    new_pc = pc.copy()
    heatmap = np.ones((new_pc.shape[0], 1), dtype=np.float32)
    for s_i, s in enumerate(other_sim):
        heatmap[other_ids[s_i]] = s
    return heatmap.flatten()


def visualize_masked_map_3d(pc: np.ndarray, mask: np.ndarray, rgb: np.ndarray, transparency: float = 0.5, min_height=-1.55, max_height=3.0):
    heatmap = mask.astype(np.float16)
    visualize_heatmap_3d(pc, heatmap, rgb, transparency, min_height, max_height)


def visualize_heatmap_3d(pc: np.ndarray, heatmap: np.ndarray, rgb: np.ndarray, transparency: float = 0.5, min_height=-1.55, max_height=3.0):
    grid_height = pc[:, 2] * 0.05
    grid_height_mask = np.logical_and(grid_height > min_height, grid_height < max_height)
    pc = pc[grid_height_mask, :]
    rgb = rgb[grid_height_mask, :]
    heatmap = heatmap[grid_height_mask]
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat.reshape(-1, 3)[:, ::-1].astype(np.float32)
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_3d(pc, heat_rgb)


def pool_3d_label_to_2d(mask_3d: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    mask_2d = np.zeros((gs, gs), dtype=bool)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        mask_2d[row, col] = mask_3d[i] or mask_2d[row, col]

    return mask_2d


def pool_3d_rgb_to_2d(rgb: np.ndarray, grid_pos: np.ndarray, gs: int) -> np.ndarray:
    rgb_2d = np.zeros((gs, gs, 3), dtype=np.uint8)
    height = -100 * np.ones((gs, gs), dtype=np.int32)
    for i, pos in enumerate(grid_pos):
        row, col, h = pos
        if h > height[row, col]:
            rgb_2d[row, col] = rgb[i]

    return rgb_2d


def get_heatmap_from_mask_2d(mask: np.ndarray, cell_size: float = 0.05, decay_rate: float = 0.01) -> np.ndarray:
    dists = distance_transform_edt(mask == 0) / cell_size
    tmp = np.ones_like(dists) - (dists * decay_rate)
    heatmap = np.where(tmp < 0, np.zeros_like(tmp), tmp)

    return heatmap


def visualize_rgb_map_2d(rgb: np.ndarray):
    """visualize rgb image

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
    """
    rgb = rgb.astype(np.uint8)
    bgr = rgb[:, :, ::-1]
    cv2.imshow("rgb map", bgr)
    cv2.waitKey(0)


def visualize_heatmap_2d(rgb: np.ndarray, heatmap: np.ndarray, transparency: float = 0.5):
    """visualize heatmap

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        heatmap (np.ndarray): (gs, gs) element range [0, 1] np.float32
    """
    sim_new = (heatmap * 255).astype(np.uint8)
    heat = cv2.applyColorMap(sim_new, cv2.COLORMAP_JET)
    heat = heat[:, :, ::-1].astype(np.float32)  # convert to RGB
    heat_rgb = heat * transparency + rgb * (1 - transparency)
    visualize_rgb_map_2d(heat_rgb)


def visualize_masked_map_2d(rgb: np.ndarray, mask: np.ndarray):
    """visualize masked map

    Args:
        rgb (np.ndarray): (gs, gs, 3) element range [0, 255] np.uint8
        mask (np.ndarray): (gs, gs) element range [0, 1] np.uint8
    """
    visualize_heatmap_2d(rgb, mask.astype(np.float32))

def visualize_colored_point_cloud(pc: np.ndarray, scores_max: np.ndarray, categories: list, min_height=-1.55, max_height=3.0):
    # 根据 target ID 分配颜色给点
    rgb = assign_colors_to_target_ids(scores_max, categories)
    grid_height = pc[:, 2] * 0.05
    grid_height_mask = np.logical_and(grid_height > min_height, grid_height < max_height)
    pc = pc[grid_height_mask, :]
    rgb = rgb[grid_height_mask, :]
    # 创建一个 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])

def assign_colors_to_target_ids(scores_max: np.ndarray, categories: list) -> np.ndarray:
    """
    为每个 target_id 分配高区分度颜色（colorcet glasbey）。
    scores_max: 一维/多维 int 数组，类别编号
    categories : 原始类别名字列表，会被自动补 'other'
    return     : (N,3) float RGB 颜色数组，范围 0~1
    """
    # 1. 展平并转成 Python int，确保可哈希
    scores_max = np.asarray(scores_max).ravel().astype(int)

    # 2. 给 categories 补 'other'，并保证长度足够
    max_id = scores_max.max()
    while len(categories) <= max_id:
        categories.append('other')

    # 3. 取离散色盘（glasbey 共 256 色，>100 类也能区分）
    cmap = cc.cm.glasbey_bw_minc_20
    colors = cmap(np.linspace(0, 1, max_id + 1))[:, :3]  # (max_id+1,3)

    # 4. 建立 id->color 映射并上色
    rgb_colors = colors[scores_max]

    # 5. 使用 OpenCV 画颜色图例（垂直排列，水平文字）
    unique_ids = np.unique(scores_max)
    bar_height = 30  # 每个矩形的高度（像素）
    bar_width = 100  # 矩形的宽度（像素）
    img_height = len(unique_ids) * bar_height + 50  # 总高度，留点边距
    img_width = bar_width + 50  # 总宽度，留点边距
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # 白色背景

    for i, uid in enumerate(unique_ids):
        # 颜色从 [0,1] 转换为 [0,255]，RGB 转 BGR（OpenCV 使用 BGR）
        color = (colors[uid][::-1] * 255).astype(np.uint8)  # RGB -> BGR
        # 绘制矩形
        top_left = (25, i * bar_height + 25)
        bottom_right = (25 + bar_width, i * bar_height + 25 + bar_height)
        cv2.rectangle(img, top_left, bottom_right, color.tolist(), -1)  # 填充矩形
        cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 1)  # 黑色边框

        # 添加文字
        text = categories[uid]
        text_color = (255, 255, 255) if colors[uid][:3].sum() < 1.5 else (0, 0, 0)  # 白或黑
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = top_left[0] + (bar_width - text_size[0]) // 2
        text_y = top_left[1] + (bar_height + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # 添加标题
    cv2.putText(img, "Category Colors (colorcet glasbey)", (25, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 显示图像（非阻塞）
    cv2.imshow("Category Colors", img)
    cv2.waitKey()  # 短暂等待，使窗口显示但不阻塞
    # 注意：窗口需要手动关闭，或者可以通过 cv2.destroyAllWindows() 关闭

    return rgb_colors