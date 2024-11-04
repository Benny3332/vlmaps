from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    vlmap = VLMap(config.map_config)
    data_dir = Path(config.data_paths.vlmaps_data_dir)
    print("data_dir: ",data_dir)
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    print("data_dirs: {} scene_id: {}".format(data_dirs[config.scene_id], config.scene_id))
    vlmap.create_map(data_dirs[config.scene_id])


if __name__ == "__main__":
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'create_map.png'
    #
    # with PyCallGraph(output=graphviz):
        main()
