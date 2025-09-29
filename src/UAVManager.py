import rasterio
import pandas as pd
from pathlib import Path

from .PathManager import PathManager
from .ConfigParser import ConfigParser
from .utils.zenodo_downloader import download_manager_without_token, get_version_from_session_name

class UAVManager:

    def __init__(self, cp: ConfigParser, pm: PathManager) -> None:
        
        self.cp = cp
        self.pm = pm

        self.ortho_information, self.ortho_information_by_name = {}, {}
        self.default_crs_uav = None
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
            
            ia_filepath = Path(session_path_ia, f"{session}_{self.cp.uav_segmentation_model_name}_ortho_predictions.tif")
            if not ia_filepath.exists() or not ia_filepath.is_file(): 
                print(f"Cannot find raster for session {ia_filepath}")
                return


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