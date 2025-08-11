from pathlib import Path
import hydra
from omegaconf import DictConfig
@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    print("Available data directories:\n" + 
          "\n".join([f"[{i}] {d}" for i, d in enumerate(data_dirs)]))
    print(f"choice scene: {data_dirs[config.scene_id]}")

if __name__ == "__main__":
    main()