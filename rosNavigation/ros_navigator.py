import numpy as np
from map_utils import build_visgraph_with_obs_map


class Navigator:
    def __init__(self):
        pass
    def build_visgraph(self, obstacle_map: np.ndarray, rowmin: float, colmin: float, vis: bool = False):
        self.obs_map = obstacle_map
        # 根据障碍物地图构建可视化图。
        self.visgraph = build_visgraph_with_obs_map(obstacle_map, True, vis=vis)
        self.rowmin = rowmin
        self.colmin = colmin