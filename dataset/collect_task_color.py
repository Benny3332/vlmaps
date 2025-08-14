import os
import time
import hydra
import json
import numpy as np
from omegaconf import DictConfig
import habitat_sim
from vlmaps.utils.habitat_utils import *
from scipy.spatial.transform import Rotation as R
import cv2
from pathlib import Path

# 全局变量存储数据集
dataset = []
current_task = None
task_id_counter = 0
same_floor_objects_list = None

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

def find_closest_object_from_class(class_name, pos_hab):
    get_all_objects(sim, pos_hab)  # Dynamically update based on current position
    class_objects = get_class_objects(class_name)
    if not class_objects:
        return None, None
    dists_list = []
    for obj in class_objects:
        obj_pos = obj.aabb.center
        obj_size = obj.aabb.sizes
        # 计算2D距离（忽略Y轴）
        dist = np.linalg.norm(obj_pos[[0, 2]] - pos_hab[[0, 2]])
        dists_list.append(dist)
    ranks = np.argsort(np.array(dists_list))
    closest_obj = class_objects[ranks[0]]
    closest_dist = dists_list[ranks[0]]
    return closest_obj, closest_dist

def build_tf_matrix(position, rotation):
    """构建4x4变换矩阵"""
    # 创建旋转矩阵
    rot_matrix = R.from_quat(rotation).as_matrix()
    # 构建4x4变换矩阵
    tf_matrix = np.eye(4)
    tf_matrix[:3, :3] = rot_matrix
    tf_matrix[:3, 3] = position
    return tf_matrix.tolist()

def save_dataset(config):
    if dataset:
        output_path = os.path.join(config.vlmaps_data_dir, "object_dataset.json")
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved dataset with {len(dataset)} tasks to {output_path}")

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
    dataset_dir = Path(config.vlmaps_data_dir)
    for scene_name in config.scene_names:
        print(f"Processing scene: {scene_name}")
        test_scene = os.path.join(config.habitat_scene_dir, scene_name, scene_name + ".glb")
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
            "img_save_dir": str(dataset_dir),
        }
        cfg = make_cfg(sim_setting)
        sim = habitat_sim.Simulator(cfg)
        agent = sim.initialize_agent(sim_setting["default_agent"])
        # 设置初始位置
        agent_state = habitat_sim.AgentState()
        random_pt = sim.pathfinder.get_random_navigable_point()
        agent_state.position = random_pt
        agent_state.rotation = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数
        agent.set_state(agent_state)
        # 初始化物体列表
        get_all_objects(sim, agent_state.position)
        print("Controls:")
        print("  'r' - Record current position as task start")
        print("  'n' - Add nearest object of specified class")
        print("  'e' - End current task and save to dataset")
        print("  'q' - Quit and save dataset")
        while True:
            obs = sim.get_sensor_observations(0)
            send_rgb(obs)
            k, action = keyboard_control_fast()
            if k == -1:
                time.sleep(0.01)
                continue
            if action == "quit":  # 'q'
                break
            elif action == "record":  # 'r'
                if current_task is not None:
                    print("Warning: Ending previous task")
                    if current_task["objects_info"]:
                        dataset.append(current_task)
                        print(f"Saved task {task_id_counter} with {len(current_task['objects_info'])} objects")
                        task_id_counter += 1
                        save_dataset(config)  # Dynamic save
                    else:
                        print("Discarding empty task")
                    current_task = None
                # 记录当前状态作为新任务的开始
                current_state = agent.get_state()
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
                # 获取物体类别名称
                class_name = input("Enter object class name: ").strip()
                if not class_name:
                    print("Error: Empty class name")
                    continue
                # 查找最近的物体
                current_pos = agent.get_state().position
                closest_obj, dist = find_closest_object_from_class(class_name, current_pos)
                if closest_obj is None:
                    print(f"No objects of class '{class_name}' found")
                    continue
                # 计算半径（最大边长的一半）
                radius = max(closest_obj.aabb.sizes) / 2.0
                # 添加物体信息
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
                if not current_task["objects_info"]:
                    print("Warning: No objects in current task")
                else:
                    dataset.append(current_task)
                    print(f"Saved task {task_id_counter} with {len(current_task['objects_info'])} objects")
                    task_id_counter += 1
                    save_dataset(config)  # Dynamic save
                current_task = None
        sim.close()
    # 最终保存数据集
    save_dataset(config)
    if not dataset:
        print("No data collected")

def send_rgb(obs):
    """显示RGB图像但不保存"""
    rgb = obs["color_sensor"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("rgb", bgr)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()