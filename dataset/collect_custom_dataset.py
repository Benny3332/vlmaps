import os
import time
import hydra
import numpy as np
from omegaconf import DictConfig
import habitat_sim
from vlmaps.utils.habitat_utils import *
import cv2
d3_40_colors_rgb = np.array([
    [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
    [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
    [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
    [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
    [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
    [57, 59, 121], [82, 84, 163], [107, 110, 207], [156, 158, 222],
    [99, 121, 57], [140, 162, 82], [181, 207, 107], [206, 219, 156],
    [140, 109, 49], [189, 158, 57], [231, 186, 82], [231, 203, 148],
    [132, 60, 57], [173, 73, 74], [214, 97, 107], [231, 150, 156],
    [123, 65, 115], [165, 81, 148], [206, 109, 189], [222, 158, 214]
], dtype=np.uint8)

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="collect_dataset.yaml",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.makedirs(config.vlmaps_data_dir, exist_ok=True)
    dataset_dir = Path(config.vlmaps_data_dir)

    scene_dirs = []
    for scene_name in config.scene_names:
        id = 1
        while True:
            scene_dir = dataset_dir / f"{scene_name}_{id}"
            if not scene_dir.exists():
                break
            id += 1
        print(f"Collecting data for scene {scene_name}")
        print(f"Data will be saved at {scene_dir}")
        scene_dir.mkdir(parents=True, exist_ok=True)
        scene_dirs.append(scene_dir)

        # test_scene_dir = config.data_paths.habitat_scene_dir
        # # test_scene_dir = "/home/hcg/hcg/phd/projects/vln/data/scene_datasets/mp3d/v1/tasks/mp3d_habitat/mp3d/"
        # # img_save_dir = "/home/huang/Pictures/vln"
        # img_save_dir = "/home/huang/hcg/projects/vln/data/clip_mapping/description_videos/"
        # # img_save_dir = "/home/hcg/Pictures/vln_diff_size_images"
        # scenes_names = os.listdir(test_scene_dir)
        # SCENE_ID = 0  # random.randint(0, len(scenes_names))
        # assert SCENE_ID < len(scenes_names) - 1

        # img_save_dir += f"{scenes_names[SCENE_ID]}_1"
        # os.makedirs(img_save_dir, exist_ok=True)

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
            "img_save_dir": scene_dir,
        }

        # cfg = make_simple_cfg(sim_setting)
        cfg = make_cfg(sim_setting)

        # create a simulator instance
        sim = habitat_sim.Simulator(cfg)
        intrinsics = get_camera_intrinsics(sim, "color_sensor")
        logging.info(f"color_sensor intrinsics: {intrinsics}")
        intrinsics = get_camera_intrinsics(sim, "depth_sensor")
        logging.info(f"depth_sensor intrinsics: {intrinsics}")
        scene = sim.semantic_scene
        objs = scene.objects
        levels = scene.levels
        for level in levels:
            print(level.id, level.aabb.center, level.aabb.sizes)
            print(
                level.id, level.aabb.center[1] - level.aabb.sizes[1] / 2, level.aabb.center[1] + level.aabb.sizes[1] / 2
            )
        # for obj in objs:
        #     print(obj.id, obj.region.category.name(), obj.category.name(), obj.obb.center, obj.obb.sizes)
        obj2cls = {int(obj.id.split("_")[-1]): (obj.category.index(), obj.category.name()) for obj in scene.objects}

        # initialize the agent
        agent = sim.initialize_agent(sim_setting["default_agent"])

        agent_state = habitat_sim.AgentState()
        random_pt = sim.pathfinder.get_random_navigable_point()
        random_pt = sim.pathfinder.get_random_navigable_point()
        random_pt = sim.pathfinder.get_random_navigable_point()
        # random_pt = sim.pathfinder.get_random_navigable_point()
        # agent_state.position = np.array([1.5, height_list[np.random.randint(0, len(height_list) - 1)], 4.0])
        agent_state.position = random_pt
        # agent.set_state(agent_state)
        # agent_state = habitat_sim.AgentState()
        pose = [3.278000593185425,	3.456643581390381,	4.238160133361816,	0.0,	0.0,	0.0,	1.0]
        agent_state.position = random_pt
        agent_state.rotation = pose[3:]
        agent.set_state(agent_state)
        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

        init_agent_state = agent_state
        actions_list = []

        obs = sim.get_sensor_observations(0)
        last_action = None
        release_count = 0
        while True:
            show_rgb(obs)
            send_semantic(obs)
            k, action = keyboard_control_fast()
            # print(f"keybroad: {k}")
            if k != -1:
                if action == "stop":
                    break
                if action == "record":
                    init_agent_state = sim.get_agent(0).get_state()
                    actions_list = []
                    continue
                last_action = action
                release_count = 0
            else:
                if last_action is None:
                    time.sleep(0.01)
                    continue
                else:
                    release_count += 1
                    if release_count > 1:
                        print("stop after release")
                        last_action = None
                        release_count = 0
                        continue
                    action = last_action

            obs = sim.step(action)
            actions_list.append(action)

        actions_list = [x for x in actions_list if x != "pause"]

        agent_states = []
        agent.set_state(init_agent_state)
        obs = sim.get_sensor_observations(0)
        root_save_dir = sim_setting["img_save_dir"]
        save_obs(root_save_dir, sim_setting, obs, 0, obj2cls)
        # save_state(root_save_dir, sim_setting, agent.get_state(), 0)
        agent_states.append(agent.get_state())

        print(f"saving frame 0/{len(actions_list) + 1}...")

        for action_i, action in enumerate(actions_list):
            obs = sim.step(action)
            agent = sim.get_agent(0)
            print(f"saving frame {action_i + 1}/{len(actions_list) + 1}...")
            save_obs(root_save_dir, sim_setting, obs, action_i + 1, obj2cls)
            agent_states.append(agent.get_state())
        save_states(root_save_dir, agent_states)

def get_camera_intrinsics(sim, sensor_name):
    # 获取渲染相机
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera
    # 获取投影矩阵
    projection_matrix = render_camera.projection_matrix
    # 获取视口大小（分辨率）
    viewport_size = render_camera.viewport
    # 计算内参
    fx = projection_matrix[0, 0] * viewport_size[0] / 2.0
    fy = projection_matrix[1, 1] * viewport_size[1] / 2.0
    cx = (projection_matrix[2, 0] + 1.0) * viewport_size[0] / 2.0
    cy = (projection_matrix[2, 1] + 1.0) * viewport_size[1] / 2.0
    intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return intrinsics

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

if __name__ == "__main__":

    main()
