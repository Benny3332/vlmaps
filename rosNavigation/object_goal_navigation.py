import os
import hydra
from pathlib import Path

from omegaconf import DictConfig
from load_scenc import HabitatLanguageRobot
from gml_floor_4_lab import gml4cat


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="object_goal_navigation_cfg",
)
def main(config: DictConfig) -> None:
    # 设置环境变量，关闭日志输出
    os.environ["MAGNUM_LOG"] = "quiet"
    # 获取数据目录
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    # 设置地图路径，设置地图和相机参数
    robot = HabitatLanguageRobot(config)
    # 获取场景ID列表
    scene_ids = []
    if isinstance(config.scene_id, int):
        scene_ids.append(config.scene_id)
    else:
        scene_ids = config.scene_id
    # 遍历场景ID列表
    for scene_i, scene_id in enumerate(scene_ids):
        # 设置场景
        robot.setup_scene(scene_id)
        robot.map.init_categories(gml4cat.copy())
        for cat in ["officeChair", "table"]:
            print(f"Navigating to category {cat}")
            robot.set_nav_curr_pose(3,3, 30)
            obj_pos = robot.find_location_to_object(cat)
            print(f"Found {cat} at position: {obj_pos}")


if __name__ == "__main__":
    main()