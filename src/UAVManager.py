import rasterio
from rasterio.enums import Compression

import numpy as np
from pathlib import Path

from .PathManager import PathManager
from .ConfigParser import ConfigParser
from .utils.zenodo_downloader import download_manager_without_token, get_version_from_session_name

class UAVManager:

    def __init__(self, cp: ConfigParser, pm: PathManager) -> None:
        
        self.cp = cp
        self.pm = pm

        self.ia_filepath, self.ia_filepath_4band = Path(), Path()
        self.setup()

    
    def setup(self) -> None:
        
        print("\n\n------ [UAV - Download IA] ------\n")
        # Download sessions
        for session in self.cp.uav_sessions:
            session_path = Path(self.pm.uav_sessions_folder, session)
            self.setup_session_uav(session_path)

            session_path_ia = Path(session_path, "PROCESSED_DATA", "IA")
            if not session_path_ia.exists() or not session_path_ia.is_dir() or len(list(session_path_ia.iterdir())) == 0: 
                print(f"Cannot find raster folder for session {session_path_ia}")
                return
            
            self.ia_filepath = Path(session_path_ia, f"{session}_{self.cp.uav_segmentation_model_name}_ortho_predictions.tif")
            if not self.ia_filepath.exists() or not self.ia_filepath.is_file(): 
                print(f"Cannot find raster for session {self.ia_filepath}")
                return
            
            
            self.ia_filepath_4band = Path(session_path_ia, f"{session}_{self.cp.uav_segmentation_model_name}_ortho_predictions_4band.tif")
            self.generate_4band_raster()


    def setup_session_uav(self, session_path: Path) -> None:
        """ Download only the IA folder for an UAV session."""
        path_ia_session = Path(session_path, "PROCESSED_DATA", "IA")

        if path_ia_session.exists() and len(list(path_ia_session.iterdir())) != 0:
            print(f"Don't download the session {session_path.name}, ia folder already exists")
            return
        
        version_json = get_version_from_session_name(session_path.name)
        if version_json == {} or "files" not in version_json:
            raise FileNotFoundError(f"Version not found for {session_path.name}")
        
        list_files = [d for d in version_json["files"] if d["key"] in ["PROCESSED_DATA_IA.zip"]]

        # Continue if no files to download due to access_right not open.
        if len(list_files) == 0 and version_json["metadata"]["access_right"] != "open":
            print("[WARNING] No files to download, version is not open.")
            return
            
        # In case we get a conceptrecid from the user, get doi
        doi = version_json["id"]

        download_manager_without_token(list_files, session_path, doi)
    

    def generate_4band_raster(self) -> None:
        """ Transform the SegForcoral output into a 4 values raster."""

        print("Transform the raster into a 4 values raster without tabular. ")

        if self.ia_filepath_4band.exists():
            self.ia_filepath_4band.unlink()
        
        # Transform Tabular into other corals and shift index.
        with rasterio.open(self.ia_filepath) as src:
            profile = src.profile
            data = src.read(1)  # read first band
        
        # Merge Tabular into Other Corals
        data[data == 2] = 4
        new_data = np.zeros_like(data, dtype=rasterio.uint8)

        new_data[data == 1] = 1  # Acropora Branching
        new_data[data == 3] = 2  # Non-acropora Massive
        new_data[data == 4] = 3  # Other Corals + Acropora Tabular
        new_data[data == 5] = 4  # Sand

        new_profile = {
             "driver":"GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "dtype": np.uint8,
            "count": 1,
            "crs": profile.get("crs"),
            "transform": profile.get("transform"),
            "compress": "LZW",
            "nodata": 0,
        }

        with rasterio.open(self.ia_filepath_4band, "w", **new_profile) as dst:
            dst.write(new_data, 1)