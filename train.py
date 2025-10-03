from argparse import Namespace, ArgumentParser

from src.ConfigParser import ConfigParser
from src.PathManager import PathManager
from src.UAVManager import UAVManager
from src.TileManager import TileManager
from src.IGNManager import IGNManager
from src.utils.lib_tools import print_header
from src.utils.training_step import TrainingStep

from src.training.main import main_launch_training
from inference import main_raster as inference_main

def parse_args() -> Namespace:

    parser = ArgumentParser(prog="Ign-upscaling training", description="Workflow to train on IGN tiles using UAV predictions.")

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

    # Initialize ign tile manager.
    ign_manager = IGNManager(cp, pm)

    # Download uav orthophoto.
    uav_manager = UAVManager(cp, pm)

    # Extract tiles and annotations.
    tile_manager = TileManager(cp, pm)
    tile_manager.create_tiles_and_annotations(uav_manager, ign_manager)
    tile_manager.convert_tiff_to_png(pm.coarse_cropped_ortho_tif_folder, pm.coarse_train_images_folder)
    tile_manager.convert_tiff_to_png(pm.coarse_annotation_tif_folder, pm.coarse_train_annotation_folder)
    tile_manager.validate_annotations_pngs()

    # First training.
    if cp.model_path_coarse == None:
        first_model_path = main_launch_training(cp, pm.coarse_train_folder, TrainingStep.COARSE)
    else:
        first_model_path = cp.model_path_coarse

    # Perform inference
    if not pm.ign_prediction_inference_raster_folder.exists() or len(list(pm.ign_prediction_inference_raster_folder.iterdir())) == 0:
        inference_args = Namespace(
            enable_folder=True, enable_session=False, enable_csv=False, 
            path_folder=pm.ign_useful_data, path_session=None, path_csv_file=None, 
            path_segmentation_model=first_model_path, 
            path_geojson=['./configs/emprise_lagoon.geojson'], 
            horizontal_overlap=0.75, 
            vertical_overlap=0.75, 
            tile_size=256, 
            path_output=pm.output_path, 
            index_start='0', 
            clean=True, 
            max_pixels_by_slice_of_rasters=800000000,
            regroup_all_prediction=True
        )
        inference_main(inference_args)
    else:
        print("\n\n------ [INFERENCE - Predictions rasters already exists] ------\n")

    # Regroup all predictions into one big file to apply seagrass annotation.
    ign_manager.regroup_inference_pred_into_one_file_by_year()

    # Apply Seagrass annotation on bigtile.



    # Split The tile

    # Retrain
        # First training.
    # if cp.model_path_refine == None:
    #     second_model_path = main_launch_training(cp, pm.refine_train_folder, TrainingStep.REFINE)
    # else:
    #     second_model_path = cp.model_path_refine

    # Perform evaluation


if __name__ == "__main__":
    opt = parse_args()
    main(opt)