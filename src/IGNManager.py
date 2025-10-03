import requests
import geopandas as gpd
from pathlib import Path

from .ConfigParser import ConfigParser
from .PathManager import PathManager
from .utils.tiles_tools import clip_raster_to_polygons, incremental_merge_tifs_windowed
from .utils.zenodo_downloader import file_downloader, extract_7z_parts

class IGNManager:

    def __init__(self, cp: ConfigParser, pm: PathManager):
        self.cp = cp
        self.pm = pm

        self.setup()

    
    def setup(self) -> None:
        
        if not self.pm.ign_useful_data.exists() or len(list(self.pm.ign_useful_data.iterdir())) == 0:
            self.download_and_uncompress()
            self.convert_and_cut_ign_data()


    def get_orthophoto_by_place(self, place: str) -> list[Path]:
        """ Retrieve only orthophoto at correct place. """
        
        ign_code = self.cp.match_place_with_ign_code.get(place, None)
        if ign_code == None:
            raise NameError(f"Please fill the config file to get a match with the ign code : {ign_code}")
        
        orthos = [ortho for ortho in self.pm.ign_useful_data.iterdir() if ign_code in ortho.name]

        return orthos


    def download_and_uncompress(self) -> None:
        """ From IGN website, we download and uncompress the data. """

        print("\n\n------ [IGN - Download part from IGN BD ORTHO] ------\n")

        # Download ressources on IGN BD Ortho.
        for url in self.cp.ign_link_files_parts:
            output_file = Path(self.pm.ign_raw_data, Path(url).name)

            # Get final size. 
            res = requests.head(url)
            expected_file_size = int(res.headers.get("content-length"))

            print(f"\nWorking with: {output_file}")
            if output_file.exists() and expected_file_size == output_file.stat().st_size:
                continue
   
            file_downloader(url, output_file)

            # Retry is filesize is different.
            while expected_file_size != output_file.stat().st_size:
                print(f"[WARNING] Filesize different expected {expected_file_size} got {output_file.stat().st_size}. We retry.")
                output_file.unlink()
                file_downloader(url, output_file)

        print("\n\n------ [IGN - Uncompress] ------\n")
        # Uncompress.
        for file in self.pm.ign_raw_data.iterdir():
            if not file.is_file() or file.suffix != ".001": continue

            print(f"Uncompress {file.stem.replace(".7z", "")}")
            
            extract_7z_parts(file, self.pm.ign_raw_data)


    def convert_and_cut_ign_data(self) -> None:
        """ Extract, convert and move data to be exploitable. """

        print("\n\n------ [IGN - Convert, crop and move useful part.] ------\n")
        print("This process take some time.")

        footprint_to_keep = [gpd.read_file(file) for file in self.cp.ign_useful_surface]

        for folder in self.pm.ign_raw_data.iterdir():
            if not folder.is_dir(): continue # Iter only on folder.

            # Go to the folder where all tiles are stored.
            sub_folder = [sb for sb in Path(folder, "ORTHOHR").iterdir() if "1" == sb.name[0]][0]
            sub_folder = [file for file in sub_folder.iterdir() if file.is_dir()][0]
            
            for file in sub_folder.iterdir():
                if file.suffix.lower() not in [".jp2"]: continue
                
                ign_code = "-".join(file.name.split("-")[2:4]) 
                if ign_code not in self.cp.ign_layer_to_keep: continue
                
                output_file = Path(self.pm.ign_useful_data, f"{file.stem}.tif")
                clip_raster_to_polygons(file, output_file, footprint_to_keep)
    
    
    def regroup_inference_pred_into_one_file_by_year(self) -> None:
        """ Regroup all predictions into one for each year. """

        print("\n\n------ [IGN - Merge all predictions file into one by year.] ------\n")

        years = list(set([file.name.split("-")[1] for file in self.pm.ign_prediction_inference_raster_folder.iterdir()]))

        for year in years:
            files_to_group = [file for file in self.pm.ign_prediction_inference_raster_folder.iterdir() if file.name.split("-")[1] == year]
            output_file = Path(self.pm.ign_regroup_prediction, f"big_tiff_{year}.tif")

            incremental_merge_tifs_windowed(files_to_group, output_file, 256)


    def add_seagrass_annotation(self) -> None:
        """Add seagrass annotation. """

        print("\n\n------ [IGN - Add seagrass annotation to the merged tile.] ------\n")

        