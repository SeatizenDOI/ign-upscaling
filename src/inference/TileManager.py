import numpy as np
from tqdm import tqdm
import geopandas as gpd
from pathlib import Path
from argparse import Namespace
from shapely.geometry import box
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

import rasterio
from rasterio.windows import Window
from rasterio.transform import array_bounds

from ..utils.tiles_tools import convert_one_tiff_to_png
from .PathRasterManager import PathRasterManager

NUM_WORKERS = max(1, cpu_count() - 2)  # Use available CPU cores, leaving some free

class TileManager:

    def __init__(self, opt: Namespace):
        self.opt = opt
        self.tile_size, self.hs, self.vs = 0, 0, 0
        self.geojson_datas = []

        self.setup()

    def setup(self) -> None:
        self.tile_size = self.opt.tile_size
        self.hs = int(self.tile_size * (1 - self.opt.horizontal_overlap)) # Horizontal step.
        self.vs = int(self.tile_size * (1 - self.opt.vertical_overlap)) # Vertical step.

        for geojson_str in self.opt.path_geojson:
            geojson_path = Path(geojson_str)
            if not geojson_path.exists() or not geojson_path.is_file(): continue
            self.geojson_datas.append(gpd.read_file(geojson_path)) 
            
        if len(self.geojson_datas) == 0:
            print("[WARNING] GeoJSON data - We don't crop the ortho with the geojson data due to no data.")


    def split_ortho_into_tiles(self, path_manager: PathRasterManager) -> None:
        print("*\t Splitting ortho into tiles.")

        with rasterio.open(path_manager.raster_path) as ortho:
            
            tile_coords = [
                (
                    path_manager, 
                    ortho.width - self.tile_size if x + self.tile_size >= ortho.width else x, 
                    ortho.height - self.tile_size if y + self.tile_size >= ortho.height else y
                )
                for x in range(0, ortho.width, self.hs) 
                for y in range(0, ortho.height, self.vs)
            ]
            
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(self.extract_one_tile, arg): arg for arg in tile_coords}
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    future.result()
                except Exception as e:
                    print(f" Worker crashed on {futures[future]}: {e}")


    def extract_one_tile(self, args: tuple[PathRasterManager, int, int]) -> tuple[str, bool, str | None]:
        path_manager, tile_x, tile_y = args
        orthoname = path_manager.raster_path.stem.replace("_ortho", "")

        try:
            with rasterio.open(path_manager.raster_path) as ortho:
                raster_crs = ortho.crs.to_string()
                window = Window(tile_x, tile_y, self.tile_size, self.tile_size)
                tile_transform = rasterio.windows.transform(window, ortho.transform)

                if len(self.geojson_datas) != 0:
                    tile_bounds = box(*array_bounds(self.tile_size, self.tile_size, tile_transform))
                    for poly_gdf in self.geojson_datas:
                        if poly_gdf.crs != raster_crs:
                            poly_gdf = poly_gdf.to_crs(raster_crs)
                        if poly_gdf.intersects(tile_bounds).any():
                            break
                    else:
                        return (f"{orthoname}_{tile_x}_{tile_y}", True, "No intersection")

                tile_ortho = ortho.read(window=window)
                greyscale_tile = np.sum(tile_ortho, axis=0) / 3

                percentage_black_pixel = np.sum(greyscale_tile == 0) * 100 / self.tile_size**2
                if percentage_black_pixel > 5:
                    return (f"{orthoname}_{tile_x}_{tile_y}", True, "Too black")

                percentage_white_pixel = np.sum(greyscale_tile == 255) * 100 / self.tile_size**2
                if percentage_white_pixel > 10:
                    return (f"{orthoname}_{tile_x}_{tile_y}", True, "Too white")

                tile_filename = f"{orthoname}_{tile_x}_{tile_y}.tif"
                tile_output_path = Path(path_manager.cropped_ortho_folder, tile_filename)

                tile_meta = ortho.meta.copy()
                tile_meta.update({
                    "height": self.tile_size,
                    "width": self.tile_size,
                    "transform": tile_transform
                })

                with rasterio.open(tile_output_path, "w", **tile_meta) as dest:
                    dest.write(tile_ortho)

            return (f"{orthoname}_{tile_x}_{tile_y}", True, None)

        except Exception as e:
            return (f"{orthoname}_{tile_x}_{tile_y}", False, str(e))


    
    def convert_tiff_tiles_into_png(self, path_manager: PathRasterManager) -> None:
        print("*\t Convert ortho tiff tiles into png files.")
        filepaths = [(filepath, path_manager.cropped_ortho_img_folder) for filepath in path_manager.cropped_ortho_folder.iterdir()]

        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(convert_one_tiff_to_png, filepaths, ), total=len(filepaths), desc=f"Processing {path_manager.cropped_ortho_folder.name}"))
