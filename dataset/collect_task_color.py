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

# 全局变量存储数据集
dataset = []
current_task = None
task_id_counter = 0
same_floor_objects_list = None

def keyboard_control_fast():
    k = cv2.waitKey(1)
    action = None
    if k == ord("a"):
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("w"):
        action = "move_forward"
    elif k == ord("r"):  # 开始记录任务
        action = "record"
    elif k == ord("n"):  # 添加物体
        action = "add_object"
    elif k == ord("e"):  # 结束任务
        action = "end_task"
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
    
    for scene_name in config.scene_names:
        print(f"Processing scene: {scene_name}")
        test_scene = os.path.join(config.habitat_scene_dir, scene_name, scene_name + ".glb")
        
        # 模拟器配置
        sim_setting = {
            "scene": test_scene,
            "default_agent": 0,
            "sensor_height": 1.5,
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": True,
            "lidar_sensor": True,
            "move_forward": 0.1,
            "turn_left": 5,
            "turn_right": 5,
            "width": 1080,
            "height": 720,
            "enable_physics": False,
            "seed": 42,
            "lidar_fov": 360,
            "depth_img_for_lidar_n": 20,
            "img_save_dir": config.vlmaps_data_dir,
        }
        
        cfg = make_cfg(sim_setting)
        sim = habitat_sim.Simulator(cfg)
        agent = sim.initialize_agent(sim_setting["default_agent"])
        agent_state = habitat_sim.AgentState()
        random_pt = sim.pathfinder.get_random_navigable_point()
        random_pt = sim.pathfinder.get_random_navigable_point()
        random_pt = sim.pathfinder.get_random_navigable_point()
        # random_pt = sim.pathfinder.get_random_navigable_point()
        # agent_state.position = np.array([1.5, height_list[np.random.randint(0, len(height_list) - 1)], 4.0])
        agent_state.position = random_pt
        agent.set_state(agent_state)

        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

        obs = sim.get_sensor_observations(0)
        get_all_objects(sim, agent_state.position)  # 初始化物体列表
        
        print("Controls:")
        print("  'r' - Record current position as task start")
        print("  'n' - Add nearest object of specified class")
        print("  'e' - End current task and save to dataset")
        print("  'q' - Quit and save dataset")
        print("  'w' - Move forward")
        print("  'a' - Turn left")
        print("  'd' - Turn right")
        
        while True:
            send_rgb(obs)
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
                closest_obj, dist = find_closest_object_from_class(sim, class_name, current_pos)
                
                if closest_obj is None:
                    print(f"No objects of class '{class_name}' found")
                    continue
                    
                # 计算半径（最大边长的一半）
                radius = max(closest_obj.aabb.sizes) / 2.0
                obj_info = {
                    "name": class_name,
                    "object_id": closest_obj.id,
                    "color": "",
                    "color value": "",
                    "position": closest_obj.aabb.center.tolist(),
                    "radius": radius
                }
                current_task["objects_info"].append(obj_info)
                print(f"Added {class_name} at {closest_obj.aabb.center} (radius: {radius:.2f})")
                
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

def send_rgb(obs):
    """显示RGB图像但不保存"""
    rgb = obs["color_sensor"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("rgb", bgr)

if __name__ == "__main__":
    main()
