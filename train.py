from argparse import Namespace, ArgumentParser

from src.ConfigParser import ConfigParser
from src.PathManager import PathManager
from src.UAVManager import UAVManager
from src.TileManager import TileManager
from src.IGNManager import IGNManager
from src.utils.lib_tools import print_header
from src.utils.training_step import TrainingStep

from src.training.main import main_launch_training

def parse_args() -> Namespace:

    parser = ArgumentParser(prog="The point is the mask training", description="Workflow to WSSS training on UAV orthophoto using ASV prediction.")

    # Config.
    parser.add_argument("-cp", "--config_path", default="./configs/config_base.json", help="Path to the config file.")
    parser.add_argument("-ep", "--env_path", default="./.env", help="Path to the env file.")
    parser.add_argument("-oe", "--only_evaluation", action="store_true", help="Only perform evalution. Skip all the rest.")

    return parser.parse_args()


def main(opt: Namespace) -> None:

    print_header()
    
    # Initialize parser.
    cp = ConfigParser(opt)

    # Initialize path manager.
    pm = PathManager(cp.output_path)
    pm.setup(cp)

    # Download uav orthophoto.
    uav_manager = UAVManager(cp, pm)

    # Initialize ign tile manager.
    ign_manager = IGNManager(cp, pm)

    # Extract tiles and annotations.
    tile_manager = TileManager(cp, pm)
    tile_manager.create_tiles_and_annotations(uav_manager, ign_manager)
    tile_manager.convert_tiff_to_png(pm.coarse_cropped_ortho_tif_folder, pm.coarse_train_images_folder)
    tile_manager.convert_tiff_to_png(pm.coarse_upsampled_annotation_tif_folder, pm.coarse_train_annotation_folder)
    tile_manager.validate_annotations_pngs()

    # First training.
    if cp.model_path_coarse == None:
        first_model_path = main_launch_training(cp, pm.coarse_train_folder, TrainingStep.COARSE)
    else:
        first_model_path = cp.model_path_coarse

    print(first_model_path)


if __name__ == "__main__":
    opt = parse_args()
    main(opt)