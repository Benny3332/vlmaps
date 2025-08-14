import os
import time
import hydra
import webcolors
import cv2
from omegaconf import DictConfig
import habitat_sim
from vlmaps.utils.habitat_utils import *

# 1. 全局窗口和鼠标回调（只初始化一次）
cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
# 回调需要的全局变量，用 dict 包一下可随时更新
_mouse_ctx = {'img_bgr': None}

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

    cv2.setMouseCallback("rgb", _on_mouse, param=_mouse_ctx)
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
        agent_state.position = pose[:3]
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
            send_rgb(obs)
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

        # agent_states = []
        # agent.set_state(init_agent_state)
        # obs = sim.get_sensor_observations(0)
        # root_save_dir = sim_setting["img_save_dir"]
        # save_obs(root_save_dir, sim_setting, obs, 0, obj2cls)
        # # save_state(root_save_dir, sim_setting, agent.get_state(), 0)
        # agent_states.append(agent.get_state())

        # print(f"saving frame 0/{len(actions_list) + 1}...")

        # for action_i, action in enumerate(actions_list):
        #     obs = sim.step(action)
        #     agent = sim.get_agent(0)
        #     print(f"saving frame {action_i + 1}/{len(actions_list) + 1}...")
        #     save_obs(root_save_dir, sim_setting, obs, action_i + 1, obj2cls)
        #     agent_states.append(agent.get_state())
        # save_states(root_save_dir, agent_states)

def _on_mouse(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    img = param['img_bgr']
    b, g, r = img[y, x]
    rgb = (r, g, b)

    # ISCC-NBS color dictionary (name: RGB tuple). Sourced from standardized centroids.
    iscc_nbs_colors = {
        "vivid-pink": (255, 181, 186),
        "strong-pink": (234, 147, 153),
        "deep-pink": (228, 113, 122),
        "light-pink": (249, 204, 202),
        "moderate-pink": (222, 165, 164),
        "dark-pink": (192, 128, 129),
        "pale-pink": (234, 216, 215),
        "grayish-pink": (196, 174, 173),
        "pinkish-white": (234, 227, 225),
        "pinkish-gray": (193, 182, 179),
        "vivid-red": (190, 0, 50),
        "strong-red": (188, 63, 74),
        "deep-red": (132, 27, 45),
        "very deep-red": (92, 9, 35),
        "moderate-red": (171, 78, 82),
        "dark-red": (114, 47, 55),
        "very dark-red": (63, 23, 40),
        "light grayish-red": (173, 136, 132),
        "grayish-red": (144, 93, 93),
        "dark grayish-red": (84, 61, 63),
        "blackish-red": (46, 29, 33),
        "reddish-gray": (143, 129, 127),
        "dark reddish-gray": (92, 80, 79),
        "reddish-black": (40, 32, 34),
        "vivid yellowish-pink": (255, 183, 165),
        "strong yellowish-pink": (249, 147, 121),
        "deep yellowish-pink": (230, 103, 33),
        "light yellowish-pink": (244, 194, 194),
        "moderate yellowish-pink": (217, 166, 169),
        "dark yellowish-pink": (196, 131, 121),
        "pale yellowish-pink": (236, 213, 197),
        "grayish yellowish-pink": (199, 173, 163),
        "brownish-pink": (226, 88, 34),
        "vivid reddish-orange": (242, 88, 34),
        "strong reddish-orange": (218, 56, 27),
        "deep reddish-orange": (203, 109, 81),
        "moderate reddish-orange": (158, 71, 50),
        "dark reddish-orange": (180, 116, 94),
        "grayish reddish-orange": (180, 116, 94),
        "strong reddish-brown": (136, 45, 23),
        "deep reddish-brown": (86, 7, 12),
        "light reddish-brown": (168, 124, 109),
        "moderate reddish-brown": (121, 68, 59),
        "dark reddish-brown": (62, 29, 30),
        "light grayish reddish-brown": (151, 127, 115),
        "grayish reddish-brown": (103, 76, 71),
        "dark grayish reddish-brown": (67, 52, 47),
        "vivid-orange": (243, 132, 0),
        "brilliant-orange": (237, 118, 14),
        "strong-orange": (237, 129, 45),
        "deep-orange": (202, 129, 39),
        "light-orange": (251, 181, 127),
        "moderate-orange": (217, 144, 88),
        "brownish-orange": (128, 70, 27),
        "strong-brown": (89, 51, 25),
        "deep-brown": (95, 25, 12),
        "light-brown": (166, 123, 91),
        "moderate-brown": (111, 78, 55),
        "dark-brown": (66, 37, 24),
        "light grayish-brown": (149, 128, 112),
        "grayish-brown": (99, 81, 71),
        "dark grayish-brown": (62, 49, 44),
        "light brownish-gray": (142, 130, 121),
        "brownish-gray": (91, 80, 79),
        "brownish-black": (40, 32, 28),
        "vivid orange-yellow": (246, 166, 0),
        "brilliant orange-yellow": (235, 168, 36),
        "strong orange-yellow": (201, 133, 0),
        "deep orange-yellow": (251, 201, 127),
        "light orange-yellow": (251, 188, 127),
        "moderate orange-yellow": (190, 138, 61),
        "dark orange-yellow": (251, 202, 127),
        "pale orange-yellow": (250, 214, 165),
        "strong yellowish-brown": (153, 101, 21),
        "deep yellowish-brown": (101, 69, 34),
        "light yellowish-brown": (193, 154, 107),
        "moderate yellowish-brown": (130, 102, 68),
        "dark yellowish-brown": (75, 54, 33),
        "light grayish yellowish-brown": (174, 155, 130),
        "grayish yellowish-brown": (126, 109, 90),
        "dark grayish yellowish-brown": (72, 60, 50),
        "vivid-yellow": (243, 195, 0),
        "brilliant-yellow": (241, 196, 15),
        "strong-yellow": (212, 175, 55),
        "deep-yellow": (217, 174, 47),
        "light-yellow": (248, 222, 126),
        "moderate-yellow": (201, 174, 93),
        "dark-yellow": (171, 145, 68),
        "pale-yellow": (243, 229, 171),
        "grayish-yellow": (194, 178, 128),
        "dark grayish-yellow": (161, 143, 96),
        "yellowish-white": (239, 232, 205),
        "yellowish-gray": (191, 184, 165),
        "light olive-brown": (150, 113, 23),
        "moderate olive-brown": (108, 84, 30),
        "dark olive-brown": (59, 49, 33),
        "vivid greenish-yellow": (220, 211, 0),
        "brilliant greenish-yellow": (233, 228, 80),
        "strong greenish-yellow": (190, 183, 46),
        "deep greenish-yellow": (155, 148, 0),
        "light greenish-yellow": (251, 198, 121),
        "moderate greenish-yellow": (185, 180, 89),
        "dark greenish-yellow": (152, 148, 62),
        "pale greenish-yellow": (251, 197, 132),
        "grayish greenish-yellow": (185, 181, 125),
        "light-olive": (134, 126, 54),
        "moderate-olive": (102, 93, 30),
        "dark-olive": (64, 61, 33),
        "light grayish-olive": (140, 135, 103),
        "grayish-olive": (91, 88, 66),
        "dark grayish-olive": (54, 53, 39),
        "light olive-gray": (138, 133, 116),
        "olive-gray": (87, 85, 76),
        "olive-black": (37, 36, 29),
        "vivid yellow-green": (141, 182, 0),
        "brilliant yellow-green": (189, 218, 87),
        "strong yellow-green": (126, 159, 46),
        "deep yellow-green": (71, 97, 46),
        "light yellow-green": (251, 220, 137),
        "moderate yellow-green": (138, 154, 91),
        "pale yellow-green": (251, 215, 183),
        "grayish yellow-green": (143, 151, 121),
        "strong olive-green": (64, 79, 0),
        "deep olive-green": (35, 54, 0),
        "moderate olive-green": (74, 93, 35),
        "dark olive-green": (43, 61, 38),
        "grayish olive-green": (81, 87, 68),
        "dark grayish olive-green": (49, 54, 43),
        "vivid yellowish-green": (39, 166, 76),
        "brilliant yellowish-green": (131, 211, 125),
        "strong yellowish-green": (126, 212, 126),
        "deep yellowish-green": (57, 150, 74),
        "very deep yellowish-green": (0, 98, 45),
        "very light yellowish-green": (182, 229, 175),
        "light yellowish-green": (147, 197, 146),
        "moderate yellowish-green": (103, 146, 103),
        "dark yellowish-green": (53, 94, 59),
        "very dark yellowish-green": (23, 54, 32),
        "vivid-green": (0, 168, 119),
        "brilliant-green": (62, 180, 137),
        "strong-green": (126, 211, 153),
        "deep-green": (0, 122, 94),
        "very light-green": (142, 209, 178),
        "light-green": (106, 171, 142),
        "moderate-green": (59, 120, 97),
        "dark-green": (27, 77, 62),
        "very dark-green": (28, 53, 45),
        "very pale-green": (199, 230, 215),
        "pale-green": (141, 163, 153),
        "grayish-green": (94, 113, 106),
        "dark grayish-green": (58, 75, 71),
        "blackish-green": (26, 36, 33),
        "greenish-white": (223, 237, 232),
        "light greenish-gray": (178, 190, 181),
        "greenish-gray": (125, 137, 132),
        "dark greenish-gray": (78, 87, 85),
        "greenish-black": (30, 35, 33),
        "vivid bluish-green": (0, 136, 130),
        "brilliant bluish-green": (0, 122, 116),
        "strong bluish-green": (127, 199, 175),
        "deep bluish-green": (0, 68, 63),
        "very light bluish-green": (150, 222, 209),
        "light bluish-green": (102, 171, 164),
        "moderate bluish-green": (49, 120, 115),
        "dark bluish-green": (0, 75, 73),
        "very dark bluish-green": (0, 42, 41),
        "vivid greenish-blue": (0, 161, 194),
        "brilliant greenish-blue": (35, 158, 186),
        "strong greenish-blue": (46, 132, 149),
        "deep greenish-blue": (19, 136, 172),
        "very light greenish-blue": (156, 209, 220),
        "light greenish-blue": (102, 168, 188),
        "moderate greenish-blue": (54, 117, 136),
        "dark greenish-blue": (0, 73, 88),
        "very dark greenish-blue": (0, 46, 59),
        "vivid-blue": (0, 161, 194),
        "brilliant-blue": (27, 103, 165),
        "strong-blue": (0, 103, 165),
        "deep-blue": (0, 65, 106),
        "very light-blue": (161, 202, 241),
        "light-blue": (112, 163, 204),
        "moderate-blue": (67, 107, 149),
        "dark-blue": (0, 48, 78),
        "very pale-blue": (188, 212, 230),
        "pale-blue": (145, 163, 176),
        "grayish-blue": (83, 104, 120),
        "dark grayish-blue": (54, 69, 79),
        "blackish-blue": (32, 40, 48),
        "bluish-white": (233, 233, 237),
        "light bluish-gray": (180, 188, 192),
        "bluish-gray": (129, 135, 139),
        "dark bluish-gray": (81, 88, 94),
        "bluish-black": (36, 41, 47),
        "vivid purplish-blue": (48, 38, 122),
        "brilliant purplish-blue": (108, 121, 184),
        "strong purplish-blue": (84, 90, 167),
        "deep purplish-blue": (39, 36, 88),
        "very light purplish-blue": (179, 188, 226),
        "light purplish-blue": (135, 145, 191),
        "moderate purplish-blue": (78, 81, 128),
        "dark purplish-blue": (37, 36, 64),
        "very pale purplish-blue": (192, 200, 225),
        "pale purplish-blue": (140, 146, 172),
        "grayish purplish-blue": (76, 81, 109),
        "vivid-violet": (144, 101, 202),
        "brilliant-violet": (124, 78, 158),
        "strong-violet": (96, 78, 151),
        "deep-violet": (50, 23, 77),
        "very light-violet": (220, 176, 248),
        "light-violet": (140, 130, 182),
        "moderate-violet": (96, 78, 129),
        "dark-violet": (47, 33, 64),
        "very pale-violet": (208, 198, 239),
        "pale-violet": (150, 144, 171),
        "grayish-violet": (88, 78, 105),
        "vivid-purple": (154, 78, 174),
        "brilliant-purple": (211, 153, 230),
        "strong-purple": (135, 86, 146),
        "deep-purple": (96, 47, 107),
        "very deep-purple": (64, 26, 76),
        "very light-purple": (213, 186, 219),
        "light-purple": (182, 149, 192),
        "moderate-purple": (134, 96, 142),
        "dark-purple": (86, 60, 92),
        "very dark-purple": (48, 25, 52),
        "very pale-purple": (214, 202, 221),
        "pale-purple": (170, 152, 169),
        "grayish-purple": (121, 104, 120),
        "dark grayish-purple": (80, 64, 77),
        "blackish-purple": (41, 30, 41),
        "purplish-white": (232, 227, 229),
        "light purplish-gray": (191, 185, 189),
        "purplish-gray": (139, 133, 137),
        "dark purplish-gray": (93, 85, 91),
        "purplish-black": (36, 30, 41),
        "vivid reddish-purple": (135, 0, 116),
        "strong reddish-purple": (158, 79, 136),
        "deep reddish-purple": (112, 73, 99),
        "very deep reddish-purple": (84, 25, 78),
        "light reddish-purple": (183, 132, 167),
        "moderate reddish-purple": (145, 92, 131),
        "dark reddish-purple": (93, 59, 84),
        "very dark reddish-purple": (52, 27, 60),
        "pale reddish-purple": (170, 138, 158),
        "grayish reddish-purple": (131, 100, 121),
        "brilliant purplish-pink": (252, 200, 214),
        "strong purplish-pink": (230, 143, 172),
        "deep purplish-pink": (222, 111, 161),
        "light purplish-pink": (239, 187, 204),
        "moderate purplish-pink": (213, 151, 174),
        "dark purplish-pink": (193, 126, 145),
        "pale purplish-pink": (232, 204, 215),
        "grayish purplish-pink": (195, 166, 177),
        "vivid purplish-red": (179, 70, 108),
        "strong purplish-red": (179, 55, 113),
        "deep purplish-red": (120, 52, 75),
        "very deep purplish-red": (84, 19, 59),
        "moderate purplish-red": (168, 81, 110),
        "dark purplish-red": (103, 49, 71),
        "very dark purplish-red": (56, 21, 44),
        "light grayish purplish-red": (175, 134, 142),
        "grayish purplish-red": (145, 95, 109),
        "white": (242, 243, 244),
        "light-gray": (185, 184, 181),
        "medium-gray": (85, 85, 85),
        "dark-gray": (34, 34, 34),
        "black": (0, 0, 0),
    }

    def closest_iscc_nbs_color(rgb):
        min_distance = float('inf')
        closest_name = "unknown"
        for name, target_rgb in iscc_nbs_colors.items():
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb, target_rgb)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        return closest_name

    # Get ISCC-NBS name
    iscc_name = closest_iscc_nbs_color(rgb)

    # Fallback to CSS3 if needed (optional; remove if not wanted)
    try:
        css_name = webcolors.rgb_to_name(rgb)
    except ValueError:
        css_name = "unknown"

    print(f"[RGB] ({x},{y}) {rgb} -> ISCC-NBS: {iscc_name} (CSS3 fallback: {css_name})")


def send_rgb(obs):
    """
    把 habitat 的 RGB 帧送到窗口，并更新鼠标回调要用的最新帧。
    不阻塞，瞬间返回。
    """
    bgr = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_RGB2BGR)
    _mouse_ctx['img_bgr'] = bgr          # 供回调使用
    cv2.imshow("rgb", bgr)

if __name__ == "__main__":

    main()
