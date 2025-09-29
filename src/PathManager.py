import shutil
from pathlib import Path

from .ConfigParser import ConfigParser

class PathManager:

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path

        self.uav_folder = Path(self.output_path, "uav")
        self.uav_sessions_folder = Path(self.uav_folder, "sessions")


    def setup(self, cp: ConfigParser) -> None:
        print("------ [CLEANING] ------")
        if cp.clean_uav_session() and self.uav_sessions_folder.exists():
            print(f"* Delete {self.uav_sessions_folder}")
            shutil.rmtree(self.uav_sessions_folder)

        print(f"* Create all subfolders.")
        self.uav_sessions_folder.mkdir(exist_ok=True, parents=True)
