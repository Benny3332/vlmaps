import os
import time
import hydra
import json
import numpy as np
from omegaconf import DictConfig
import habitat_sim
from habitat_sim.utils.common import quat_from_magnum, quat_to_magnum
from vlmaps.utils.habitat_utils import *
import cv2
from pathlib import Path
import magnum as mn
from sklearn.cluster import KMeans
from PIL import Image
from color_text_dict import (iscc_nbs_colors, d3_40_colors_rgb_list)

# 全局变量存储数据集
dataset = []
current_task = None
task_id_counter = 0
same_floor_objects_list = None

d3_40_colors_rgb = np.array(d3_40_colors_rgb_list, dtype=np.uint8)

def keyboard_control_fast():
    k = cv2.waitKey(1)
    action = None
    if k == ord("a"):
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("w"):
        action = "move_forward"
    elif k == ord("s"): 
        action = "move_backward"
    elif k == ord("i"): 
        action = "look_up"
    elif k == ord("k"):
        action = "look_down"
    elif k == ord("r"):  # 开始记录任务
        action = "record"
    elif k == ord("n"):  # 添加物体
        action = "add_object"
    elif k == ord("m"):  # 移动后捕获颜色
        action = "move_and_capture"
    elif k == ord("e"):  # 结束任务
        action = "end_task"
    elif k == ord("l"):  # 识别当前画面中的物体
        action = "list_objects"
    elif k == ord("q"):  # 退出程序
        action = "quit"
    return k, action

def get_all_objects(sim, position, h_thres=2.0):
    global same_floor_objects_list
    scene = sim.semantic_scene
    same_floor_objects_list = get_position_floor_objects(scene, position, h_thres)

def get_position_floor_objects(semantic_scene, position, h_thres, concept_type="object"):
    if concept_type == "object":
        objects = semantic_scene.objects
    else:
        objects = semantic_scene.regions
    same_floor_obj_list = []
    for obj in objects:
        if concept_type == "object":
            obj_h = obj.obb.center[1]
        else:
            obj_h = obj.aabb.center[1]
        if abs(obj_h - position[1]) < h_thres:
            same_floor_obj_list.append(obj)
    return same_floor_obj_list

def get_class_objects(class_name):
    global same_floor_objects_list
    if same_floor_objects_list is None:
        raise RuntimeError("Call get_all_objects() before calling get_class_objects()")
    return [x for x in same_floor_objects_list if x.category.name() == class_name]

def find_closest_object_from_class(sim, class_name, pos_hab):
    get_all_objects(sim, pos_hab)  # 基于当前位置动态更新
    class_objects = get_class_objects(class_name)
    if not class_objects:
        return None, None
    dists_list = []
    for obj in class_objects:
        obj_pos = obj.aabb.center
        # 计算2D距离（忽略Y轴）
        dist = np.linalg.norm(obj_pos[[0, 2]] - pos_hab[[0, 2]])
        dists_list.append(dist)
    min_idx = np.argmin(dists_list)
    return class_objects[min_idx], dists_list[min_idx]

def build_tf_matrix(position, rotation):
    """构建4x4变换矩阵"""
    
    # 处理不同类型的四元数
    if isinstance(rotation, mn.Quaternion):
        # Magnum四元数，直接使用
        rot_quat = rotation
    elif hasattr(rotation, 'x') and hasattr(rotation, 'y') and hasattr(rotation, 'z') and hasattr(rotation, 'w'):
        # quaternion.quaternion类型 (x, y, z, w)
        quat_arr = np.array([
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w
        ], dtype=np.float64)
        # 直接创建Magnum四元数 (vector, scalar)
        rot_quat = mn.Quaternion(mn.Vector3(quat_arr[:3]), quat_arr[3])
    else:
        # 假设是数组形式 [x, y, z, w]
        quat_arr = np.array(rotation, dtype=np.float64)
        # 直接创建Magnum四元数 (vector, scalar)
        rot_quat = mn.Quaternion(mn.Vector3(quat_arr[:3]), quat_arr[3])
    
    # 创建4x4变换矩阵
    tf_matrix = np.eye(4)
    
    # 获取旋转矩阵
    rot_mat = rot_quat.to_matrix()
    
    # 填充旋转部分
    for i in range(3):
        for j in range(3):
            tf_matrix[i, j] = rot_mat[i][j]
    
    # 填充平移部分
    tf_matrix[0, 3] = position[0]
    tf_matrix[1, 3] = position[1]
    tf_matrix[2, 3] = position[2]
    
    return tf_matrix.tolist()


def save_dataset(config, scene_name):
    global dataset
    if not dataset:
        print(f"No tasks to save for scene {scene_name}")
        return
    
    output_dir = os.path.join(config.task_dir, scene_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "color_object_nav_dataset.json")
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Saved {len(dataset)} tasks to {output_path}")

def closest_iscc_nbs_color(rgb):
    """Find the closest ISCC-NBS color name for a given RGB value"""
    min_distance = float('inf')
    closest_name = "unknown"
    for name, target_rgb in iscc_nbs_colors.items():
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, target_rgb)) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def extract_main_colors(object_pixels, n_colors=2, dominant_threshold=0.75):
    """
    使用K-means聚类提取主色和次主色，并返回颜色名称
    
    Args:
        object_pixels: 物体像素的RGB值数组
        n_colors: 要提取的颜色数量
        dominant_threshold: 主色占比阈值，超过此值则只返回一个颜色
    
    Returns:
        tuple: (color_names, color_values)
            color_names: 颜色名称列表，如['vivid-pink','strong-pink']
            color_values: 颜色RGB值列表，如[[255,0,0], [0,255,0]]
    """
    if len(object_pixels) == 0:
        return [["unknown"]], [[[128, 128, 128]]]
    
    if object_pixels.shape[1] == 4:  # 检查是否为RGBA
        object_pixels = object_pixels[:, :3]  # 只取前3通道
 
    # 如果像素太少，直接使用平均颜色
    if len(object_pixels) < n_colors:
        avg_color = np.mean(object_pixels, axis=0).astype(int).tolist()
        color_name = closest_iscc_nbs_color(avg_color)
        return [[color_name]], [[avg_color]]
    
    try:
        # 使用K-means聚类
        n_clusters = min(n_colors, len(object_pixels))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(object_pixels)
        
        cluster_centers = kmeans.cluster_centers_.astype(int)
        cluster_labels, counts = np.unique(kmeans.labels_, return_counts=True)
        
        # 计算每个聚类的占比
        proportions = counts / len(object_pixels)
        
        # 按占比排序颜色
        sorted_indices = np.argsort(proportions)[::-1]
        
        # 检查主色占比是否超过阈值
        if proportions[sorted_indices[0]] > dominant_threshold:
            # 只返回主色
            main_color_idx = sorted_indices[0]
            main_color = cluster_centers[main_color_idx].tolist()
            main_color_name = closest_iscc_nbs_color(main_color)
            return [[main_color_name]], [[main_color]]
        
        # 提取前n_colors个主要颜色
        color_names = []
        color_values = []
        for i in range(min(n_colors, len(sorted_indices))):
            idx = sorted_indices[i]
            color = cluster_centers[idx].tolist()
            color_name = closest_iscc_nbs_color(color)
            color_names.append(color_name)
            color_values.append(color)
        
        # 如果颜色数量不够，用最后一个颜色填充
        while len(color_names) < n_colors:
            color_names.append(color_names[-1] if color_names else "unknown")
            color_values.append(color_values[-1] if color_values else [128, 128, 128])
        
        return [color_names], [color_values]
        
    except Exception as e:
        print(f"KMeans聚类失败: {e}")
        avg_color = np.mean(object_pixels, axis=0).astype(int).tolist()
        color_name = closest_iscc_nbs_color(avg_color)
        return [[color_name]], [[avg_color]]

def list_visible_objects(sim, obs):
    """
    识别当前画面中的可见物体并去重
    """
    if "semantic_sensor" not in obs:
        print("Error: No semantic sensor data available")
        return
    
    semantic_obs = obs["semantic_sensor"]
    
    # 获取所有非零的语义ID（排除背景，通常为0）
    unique_semantic_ids = np.unique(semantic_obs)
    non_zero_ids = unique_semantic_ids[unique_semantic_ids != 0]
    
    if len(non_zero_ids) == 0:
        print("No objects detected in current view")
        return
    
    # 获取场景中的所有物体
    scene = sim.semantic_scene
    objects = scene.objects
    
    # 创建语义ID到物体对象的映射
    id_to_object = {}
    for obj in objects:
        # 从物体ID中提取语义ID（如"0_18_302"中的302）
        try:
            semantic_id = int(obj.id.split("_")[-1])
            id_to_object[semantic_id] = obj
        except (ValueError, IndexError):
            continue
    
    # 收集可见物体的类别名称（去重）
    visible_objects = set()
    for semantic_id in non_zero_ids:
        if semantic_id in id_to_object:
            obj = id_to_object[semantic_id]
            if obj.category and obj.category.name():
                visible_objects.add(obj.category.name())
    
    # 打印结果
    print("\nVisible objects in current view:")
    for i, obj_name in enumerate(sorted(visible_objects)):
        print(f"{i+1}. {obj_name}")
    print(f"Total unique objects: {len(visible_objects)}\n")

def get_object_colors_with_confirmation(sim, closest_obj, obs, obj2cls):
    """
    获取物体颜色信息，正确处理ID映射
    """
    # 检查语义传感器数据
    if "semantic_sensor" not in obs:
        print("Error: No semantic sensor data available")
        return None
    
    semantic_obs = obs["semantic_sensor"]
    rgb_obs = obs["color_sensor"]
    
    # 获取场景对象ID（如"0_18_302"）
    scene_object_id = closest_obj.id
    print(f"Scene object ID: {scene_object_id}")
    
    # 将场景对象ID转换为语义传感器ID
    try:
        # 从"0_18_302"提取最后一个数字作为语义ID
        # semantic_id = closest_obj.category.index()
        semantic_id = int(scene_object_id.split("_")[-1])
        print(f"Mapped to semantic ID: {semantic_id}")
    except (ValueError, IndexError) as e:
        print(f"Error parsing object ID {scene_object_id}: {e}")
        return False
    
    # 检查这个语义ID是否在当前视野中
    object_mask = (semantic_obs == semantic_id)
    visible_pixels = np.sum(object_mask)
    
    print(f"Visible pixels for semantic ID {semantic_id}: {visible_pixels}")
    
    if visible_pixels == 0:
        print(f"Semantic ID {semantic_id} is not visible in current view")
        # 显示当前视野中的语义ID
        unique_semantic_ids = np.unique(semantic_obs)
        non_zero_ids = unique_semantic_ids[unique_semantic_ids != 0]
        if len(non_zero_ids) > 0:
            print(f"Non-zero semantic IDs in view: {non_zero_ids[:10]}")
        return False
    
    # 创建可视化图像
    vis_image = rgb_obs.copy()
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    # 在掩码区域上叠加半透明红色
    overlay = vis_image_bgr.copy()
    overlay[object_mask, :3] = [0, 0, 255]  # 红色
    
    # 混合原图和overlay
    alpha = 0.3  # 透明度
    vis_image_bgr = cv2.addWeighted(vis_image_bgr, 1 - alpha, overlay, alpha, 0)
    
    # 在图像上添加文本提示
    cv2.putText(vis_image_bgr, f'Scene ID: {scene_object_id}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image_bgr, f'Semantic ID: {semantic_id}', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image_bgr, f'Pixels: {visible_pixels}', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_image_bgr, 'Press y to confirm, n to skip', (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示图像并等待用户输入
    cv2.imshow('Object Confirmation', vis_image_bgr)
    cv2.waitKey(1)  # 刷新显示
    
    print(f"Object highlighted in red. Confirm? (y/n): ")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y') or key == ord('Y'):
            cv2.destroyWindow('Object Confirmation')
            return True
        elif key == ord('n') or key == ord('N'):
            cv2.destroyWindow('Object Confirmation')
            return False

def check_scene_semantic_support(sim):
    """检查场景是否支持语义信息"""
    scene = sim.semantic_scene
    print(f"Scene objects count: {len(scene.objects) if scene.objects else 0}")
    
    if scene.objects:
        print("First few objects:")
        for i, obj in enumerate(scene.objects[:5]):
            print(f"  Object {i}: ID={obj.id}, Name={obj.category.name() if obj.category else 'No category'}")
    else:
        print("Warning: No semantic objects found in scene")
    
    # 检查观测数据
    obs = sim.get_sensor_observations()
    if "semantic_sensor" in obs:
        semantic_obs = obs["semantic_sensor"]
        print(f"Semantic sensor shape: {semantic_obs.shape}")
        print(f"Semantic sensor unique values: {np.unique(semantic_obs)[:10]}")  # 显示前10个唯一值
        print(f"Semantic sensor min/max: {semantic_obs.min()}/{semantic_obs.max()}")
    else:
        print("Warning: No semantic sensor in observations")

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="collect_dataset.yaml",
)
def main(config: DictConfig) -> None:
    global dataset, current_task, task_id_counter
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.makedirs(config.vlmaps_data_dir, exist_ok=True)

    registry = habitat_sim.registry
    move_fns = registry._mapping.get("move_fn", {})
    print("Supported actions:", list(move_fns.keys()))

    for scene_name in config.scene_names:
        print(f"Processing scene: {scene_name}")
        test_scene = os.path.join(config.habitat_scene_dir, scene_name, scene_name + ".glb")
        
        # 模拟器配置
        sim_setting = {
            "scene": test_scene,
            "default_agent": 0,
            "sensor_height": 1.4,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": True,
            "lidar_sensor": True,
            "move_forward": 0.1,
            "move_backward": 0.1,
            "turn_left": 5,
            "turn_right": 5,
            "look_up": 5.0,
            "look_down": 5.0,
            "width": 1080,
            "height": 720,
            "enable_physics": False,
            "seed": 42,
            "lidar_fov": 360,
            "depth_img_for_lidar_n": 20,
            "img_save_dir": config.vlmaps_data_dir,
            "scene_dataset_config_file": os.path.join(config.habitat_scene_dir, "mp3d.scene_dataset_config.json")
        }
        
        
        cfg = make_cfg_2(sim_setting)
        sim = habitat_sim.Simulator(cfg)
        agent = sim.initialize_agent(sim_setting["default_agent"])
        agent_state = habitat_sim.AgentState()
        pose = [8.37611198425293,	-1.2843499183654785,	7.724077224731445,	0.0,	0.0,	0.0,	1.0]
        agent_state.position = pose[:3]
        agent_state.rotation = pose[3:]
        agent.set_state(agent_state)
        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
        obs = sim.get_sensor_observations(0)
        print()
        get_all_objects(sim, agent_state.position)  # 初始化物体列表

        print("\nAvailable objects:")
        for i, obj in enumerate(same_floor_objects_list):
            print(f"ID: {obj.id}, Name: {obj.category.name()}")
        print("Please select an object by entering its ID when using 'n' command.\n")

        print("Controls:")
        print("  'r' - Record current position as task start")
        print("  'n' - Select nearest object of specified class")
        print("  'm' - Move to suitable position and capture object colors")
        print("  'e' - End current task and save to dataset")
        print("  'q' - Quit and save dataset")
        print("  'w' - Move forward")
        print("  'a' - Turn left")
        print("  'd' - Turn right")
        
        while True:
            send_rgb(obs)
            send_semantic(obs)
            k, action = keyboard_control_fast()
            
            if action is None:
                time.sleep(0.01)
                continue
                
            if action == "quit":  # 'q'
                break
                
            elif action == "record":  # 'r'
                if current_task is not None:
                    print("Warning: Ending previous unfinished task")
                    if current_task["objects_info"]:
                        dataset.append(current_task)
                        print(f"Saved task {task_id_counter} with {len(current_task['objects_info'])} objects")
                        task_id_counter += 1
                    else:
                        print("Discarding empty task")
                
                # 记录新任务
                current_state = agent.get_state()
                
                # 构建变换矩阵
                tf_matrix = build_tf_matrix(current_state.position, current_state.rotation)
                
                current_task = {
                    "task_id": task_id_counter,
                    "scene": scene_name,
                    "map_grid_size": 1000,
                    "map_cell_size": 0.05,
                    "tf_habitat": tf_matrix,
                    "objects_info": []
                }
                print(f"Started new task {task_id_counter} at position {current_state.position}")
                
            elif action == "add_object":  # 'n'
                if current_task is None:
                    print("Error: Start a task first with 'r'")
                    continue
                    
                class_name = input("Enter object class name: ").strip()
                if not class_name:
                    print("Error: Empty class name")
                    continue
                    
                current_pos = agent.get_state().position
                obs = sim.get_sensor_observations(0)  # 获取当前观测
                
                closest_obj, dist = find_closest_object_from_class(sim, class_name, current_pos)
                
                if closest_obj is None:
                    print(f"No objects of class '{class_name}' found")
                    continue
                
                # 保存物体信息，等待用户移动到合适位置后再获取颜色
                pending_object = {
                    "closest_obj": closest_obj,
                    "class_name": class_name
                }
                
                print(f"Found {class_name} (ID: {closest_obj.id})")
                print("Please move agent to suitable position and press 'm' to capture colors")

            elif action == "list_objects":  # 'l'
                obs = sim.get_sensor_observations(0)
                list_visible_objects(sim, obs)

            elif action == "move_and_capture":  # 'm'
                if current_task is None:
                    print("Error: Start a task first with 'r'")
                    continue
                    
                if 'pending_object' not in locals() or pending_object is None:
                    print("Error: No pending object. Press 'n' first to select an object")
                    continue
                
                # 获取当前观测和ID映射
                obs = sim.get_sensor_observations(0)
                obj2cls = get_obj2cls_dict(sim)  # 获取ID映射字典
                
                closest_obj = pending_object["closest_obj"]
                class_name = pending_object["class_name"]
                
                # 获取颜色信息（包含用户确认）
                confirmed = get_object_colors_with_confirmation(sim, closest_obj, obs, obj2cls)
                
                if not confirmed:
                    print("物体颜色获取被取消或物体不可见")
                    pending_object = None
                    continue
                
                # 用户确认后，提取颜色信息
                semantic_obs = obs["semantic_sensor"]
                rgb_obs = obs["color_sensor"]
                
                # 正确映射ID
                scene_object_id = closest_obj.id
                semantic_id = int(scene_object_id.split("_")[-1])
                object_mask = (semantic_obs == semantic_id)
                object_pixels = rgb_obs[object_mask]
                
                if len(object_pixels) == 0:
                    print("警告：物体像素为空，使用默认颜色")
                    color_names = [["unknown"]]
                    color_values = [[[128, 128, 128]]]
                    visible_pixels = 0
                else:
                    # 提取主色和次主色
                    color_names, color_values = extract_main_colors(object_pixels, n_colors=2)
                    visible_pixels = len(object_pixels)
                
                # 获取颜色名称列表
                color_name_list = color_names[0]
                
                # 创建物体信息字典
                obj_info = {
                    "name": class_name,
                    "object_id": scene_object_id,  # 保存原始场景ID
                    "semantic_id": semantic_id,    # 保存语义ID
                    "color": color_name_list,       # 格式: ['vivid-pink','strong-pink']
                    "color_name": f"{' and '.join(color_name_list)} {class_name}",  # 格式: "vivid-pink chair"
                    "color_value": color_values[0],  # 格式: [[主色], [次主色]]
                    "position": closest_obj.aabb.center.tolist(),
                    "radius": max(closest_obj.aabb.sizes) / 2.0,
                    "visible_pixels": visible_pixels
                }
                
                current_task["objects_info"].append(obj_info)
                
                main_color = color_values[0][0]
                main_color_name = color_name_list[0]
                secondary_color = color_values[0][1] if len(color_values[0]) > 1 else [128, 128, 128]
                secondary_color_name = color_name_list[1] if len(color_name_list) > 1 else "unknown"
                
                print(f"Added {class_name}")
                print(f"Scene ID: {scene_object_id}, Semantic ID: {semantic_id}")
                print(f"Main color: RGB{main_color} ({main_color_name})")
                print(f"Secondary color: RGB{secondary_color} ({secondary_color_name})")
                print(f"Color names: {color_name_list}")
                print(f"Color name: {obj_info['color_name']}")
                print(f"Visible pixels: {visible_pixels}")
                print(f"now task collect object num : {len(current_task['objects_info'])}")

                # 清除待处理物体
                pending_object = None
                
            elif action == "end_task":  # 'e'
                if current_task is None:
                    print("Error: No active task to end")
                    continue
                    
                if current_task["objects_info"]:
                    dataset.append(current_task)
                    print(f"Saved task {task_id_counter} with {len(current_task['objects_info'])} objects")
                    task_id_counter += 1
                else:
                    print("Task ended without any objects, discarded")
                current_task = None
                
            else:  # 移动操作 (turn_left, turn_right, move_forward)
                obs = sim.step(action)
                
                # 如果有待处理的物体，打印物体相对于当前代理位置的相对坐标
                if 'pending_object' in locals() and pending_object is not None:
                    current_pos = agent.get_state().position
                    obj_pos = pending_object["closest_obj"].aabb.center
                    relative_pos = obj_pos - current_pos
                    dist_2d = np.linalg.norm(relative_pos[[0, 2]])  # 2D距离（忽略Y轴）
                    
                    print(f"Relative coordinates to {pending_object['class_name']} (ID: {pending_object['closest_obj'].id}):")
                    print(f"  Delta X: {relative_pos[0]:.3f}, Delta Y: {relative_pos[1]:.3f}, Delta Z: {relative_pos[2]:.3f}")
                    print(f"  2D Distance (XZ plane): {dist_2d:.3f} meters")
                
        # 场景处理完成
        if current_task is not None:
            if current_task["objects_info"]:
                dataset.append(current_task)
                print(f"Saved unfinished task {task_id_counter} with {len(current_task['objects_info'])} objects")
                task_id_counter += 1
            else:
                print("Discarding empty unfinished task")
            current_task = None
            
        # 保存当前场景的数据集
        save_dataset(config, scene_name)
        
        # 重置数据集为下一个场景准备
        dataset = []
        sim.close()
    
    print(f"Finished processing all scenes. Total tasks collected: {task_id_counter}")

def send_semantic(obs):
    """可视化语义分割图像但不保存"""
    # 获取语义传感器数据
    semantic_obs = obs["semantic_sensor"]
    
    # 创建调色板图像 (P模式)
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    
    # 应用D3-40调色板 (需确保 d3_40_colors_rgb 已定义)
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    
    # 将语义ID映射到调色板索引 (取模40确保在0-39范围内)
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    
    # 转换为RGB图像并转成OpenCV格式
    semantic_rgb = np.array(semantic_img.convert("RGB"))
    semantic_bgr = cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR)
    
    # 显示图像
    cv2.imshow("semantic", semantic_bgr)


def send_rgb(obs):
    """显示RGB图像但不保存"""
    rgb = obs["color_sensor"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("rgb", bgr)

if __name__ == "__main__":
    main()
