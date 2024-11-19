from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter


from ros_map import Map
from map_utils import load_3d_map, cam_load_3d_map

class VLMap(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        super().__init__(map_config, data_dir=data_dir)
        self.scores_mat = None
        self.categories = None
    def init_categories(self, categories: List[str]) -> np.ndarray:
        vlmaps_data_dir = self.data_dir
        save_path = vlmaps_data_dir / "vlmap_cam" / "scores_mat.npy"
        print(f"Initializing categories from local store: {save_path}")
        self.scores_mat = np.load(save_path)
        return self.scores_mat
    def load_map(self, data_dir: str) -> bool:
        self._setup_paths(data_dir)
        print(self.data_dir)
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_save_path = Path(data_dir) / "vlmap" / "vlmaps.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
            ) = load_3d_map(self.map_save_path)
        elif self.map_config.pose_info.pose_type == "camera_base":
            self.map_save_path = Path(data_dir) / "vlmap_cam" / "vlmaps_cam.h5df"
            print(self.map_save_path)
            if not self.map_save_path.exists():
                assert False, "Loading VLMap failed because the file doesn't exist."
            (
                self.mapped_iter_list,
                self.grid_feat,
                self.grid_pos,
                self.weight,
                self.occupied_ids,
                self.grid_rgb,
                self.pcd_min,
                self.pcd_max,
                self.cs,
            ) = cam_load_3d_map(self.map_save_path)
        else:
            raise ValueError("Invalid pose type")

        return True