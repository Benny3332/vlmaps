from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter


from ros_map import Map


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