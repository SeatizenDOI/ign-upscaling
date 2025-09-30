import shutil
from pathlib import Path

from .ConfigParser import ConfigParser

class PathManager:

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

        self.uav_folder = Path(self.output_path, "uav")
        self.uav_sessions_folder = Path(self.uav_folder, "sessions")

        self.cropped_ortho_tif_folder = Path(self.output_path, "cropped_ortho_tif")
        self.cropped_annotation_tif_folder = Path(self.output_path, "cropped_annotation_tif")

        self.train_images_folder = Path(self.output_path, "train", "images")
        self.train_annotations_folder = Path(self.output_path, "train", "annotations")



    def setup(self, cp: ConfigParser) -> None:
        print("------ [CLEANING] ------")
        if cp.clean_uav_session() and self.uav_sessions_folder.exists():
            print(f"* Delete {self.uav_sessions_folder}")
            shutil.rmtree(self.uav_sessions_folder)

        print(f"* Create all subfolders.")
        self.uav_sessions_folder.mkdir(exist_ok=True, parents=True)

        self.cropped_ortho_tif_folder.mkdir(exist_ok=True, parents=True)
        self.cropped_annotation_tif_folder.mkdir(exist_ok=True, parents=True)
        self.train_images_folder.mkdir(exist_ok=True, parents=True)
        self.train_annotations_folder.mkdir(exist_ok=True, parents=True)

        
