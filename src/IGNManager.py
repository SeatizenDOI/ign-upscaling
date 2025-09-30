from pathlib import Path

from .ConfigParser import ConfigParser
from .PathManager import PathManager

class IGNManager:

    def __init__(self, cp: ConfigParser, pm: PathManager):
        self.cp = cp
        self.pm = pm

        self.ortho_by_place: dict[str, list[Path]] = {}
        self.setup()

    
    def setup(self) -> None:
        
        for file in self.cp.ign_parts:
            filepath = Path(file)

            if not filepath.exists():
                raise FileNotFoundError(f"File {filepath} not found")

            ign_code = "-".join(filepath.name.split("-")[2:4])
            place = self.cp.match_ign_number_with_place_name.get(ign_code, None)
            if place == None:
                raise NameError(f"Please fill the config file to get a match with the ign code : {ign_code}")
            
            if place not in self.ortho_by_place:
                self.ortho_by_place[place] = []
            self.ortho_by_place[place].append(filepath)
