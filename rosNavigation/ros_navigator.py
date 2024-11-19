import numpy as np
from map_utils import build_visgraph_with_obs_map
from typing import Tuple, List, Dict
import pyvisgraph as vg

class Navigator:
    def __init__(self):
        pass

    def build_visgraph(self, obstacle_map: np.ndarray, rowmin: float, colmin: float, vis: bool = False):
        self.obs_map = obstacle_map
        # 根据障碍物地图构建可视化图。
        self.visgraph = build_visgraph_with_obs_map(obstacle_map, True, vis=vis)
        self.rowmin = rowmin
        self.colmin = colmin

    def plan_to(
        self, start_full_map: Tuple[float, float], goal_full_map: Tuple[float, float], vis: bool = False
    ) -> List[List[float]]:
        """
        Take full map start (row, col) and full map goal (row, col) as input
        Return a list of full map path points (row, col) as the palnned path
        """
        start = self._convert_full_map_pos_to_cropped_map_pos(start_full_map)
        goal = self._convert_full_map_pos_to_cropped_map_pos(goal_full_map)
        if self._check_if_start_in_graph_obstacle(start):
            self._rebuild_visgraph(start, vis)
        paths = plan_to_pos_v2(start, goal, self.obs_map, self.visgraph, vis)
        paths = self.shift_path(paths, self.rowmin, self.colmin)
        return paths