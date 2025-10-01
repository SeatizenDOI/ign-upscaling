
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import geopandas as gpd
from pathlib import Path
from shapely.geometry import box, mapping
from multiprocessing import Pool, cpu_count

import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask

from .ConfigParser import ConfigParser
from .PathManager import PathManager
from .UAVManager import UAVManager
from .IGNManager import IGNManager
from .utils.tiles_tools import convert_one_tiff_to_png


NUM_WORKERS = max(1, cpu_count() - 2)  # Use available CPU cores, leaving some free

class TileManager:

    def __init__(self, cp: ConfigParser, pm: PathManager):
        self.cp = cp
        self.pm = pm


    def load_geojson_with_crs(self, crs: str, list_geojson_path: Path) :

        # Load the boundary IGN into the good crs for each ortho.
        gdfs = [gpd.read_file(f).to_crs(crs) for f in list_geojson_path]

        # Concat all dataframe.
        merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=crs)
        
        # Union the polygon
        return merged_gdf.union_all()


    def create_tiles_and_annotations(self, uav_manager: UAVManager, ign_manager: IGNManager) -> None:
        
        print("\n\n------ [TILE - Create IGN Tiles] ------\n")

        nb_tile_tif = len(list(self.pm.coarse_cropped_ortho_tif_folder.iterdir())) 
        nb_anno_tif = len(list(self.pm.coarse_annotation_tif_folder.iterdir()))

        if nb_anno_tif > 0 and nb_anno_tif == nb_tile_tif:
            print("We have already split the ortho into tiles.")
            return

        for annotation_path in uav_manager.annotations_files:
            
            session_name = "_".join(annotation_path.name.split("_")[0:4])
            place = "-".join(annotation_path.name.split("_")[1].split("-")[1:]) # Assume the file start with the session_name.
            ign_orthophotos_path = ign_manager.ortho_by_place.get(place, None)

            if ign_orthophotos_path == None:
                raise NameError(f"No orthophotos found for place {place}")


            for ortho_path in ign_orthophotos_path:
                print(f"Working with orthophoto {ortho_path.name}")

                with rasterio.open(ortho_path) as ortho:
                    boundary_ign = self.load_geojson_with_crs(ortho.crs, self.cp.list_boundary_ign_geojson)
                    zone_test = self.load_geojson_with_crs(ortho.crs, self.cp.list_drone_test_geojson)
 
                    tile_coords = [
                        (session_name, x, y, ortho_path, annotation_path, boundary_ign, zone_test)
                        for x in range(0, ortho.width - self.cp.tile_size + 1, self.cp.horizontal_step)
                        for y in range(0, ortho.height - self.cp.tile_size + 1, self.cp.vertical_step)
                    ]
                with Pool(NUM_WORKERS) as pool:
                    list(tqdm(pool.imap_unordered(self.process_tile, tile_coords), total=len(tile_coords)))



    def convert_tiff_to_png(self, input_dir: Path, output_dir: Path) -> None:
        print("\n\n------ [TILES - Convert tiff tiles to png] ------\n")

        if len(list(output_dir.iterdir())) > 0:
            print("We already have convert the tif into png.")
            return
        
        args = []
        for file in input_dir.iterdir():
            if file.suffix.lower() != ".tif": continue

            args.append((file, output_dir))
        
        with Pool(processes=cpu_count()) as pool:
            list(tqdm(pool.imap(convert_one_tiff_to_png, args), total=len(args), desc=f"Processing {input_dir.name}"))


    def validate_annotations_pngs(self, valid_range: tuple[int, int] = [1, 5]):
        min_valid, max_valid = valid_range

        print("\n\n------ [TILES - Validate png files] ------\n")
        cpt = 0
        for file in self.pm.coarse_train_annotation_folder.iterdir():
            if file.suffix.lower() != ".png": continue

            try:
                img = Image.open(file)
                data = np.array(img)

                if np.any(data < min_valid) or np.any(data > max_valid):
                    
                    print(f"❌ Invalid values in {file.name}, we delete it")
                    cpt += 1
                    # Delete annotation PNG
                    file.unlink()

                    # Delete annotation TIF
                    tif_path = Path(self.pm.coarse_annotation_tif_folder, f"{file.stem}.tif")
                    if tif_path.exists():
                        tif_path.unlink()

                    # Delete image PNG
                    img_png_path = Path(self.pm.coarse_train_images_folder, file.name)
                    if img_png_path.exists():
                        img_png_path.unlink()

                    # Delete image TIF
                    img_tif_path = Path(self.pm.coarse_cropped_ortho_tif_folder, f"{file.stem}.tif")
                    if img_tif_path.exists():
                        img_tif_path.unlink()

            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
                continue

        print(f"We have delete {cpt} annotations")

    def process_tile(self, args: tuple[str, int, int, Path, Path]) -> None:
        session_name, tile_x, tile_y, orthophoto_path, annotation_path, boundary_ign, zone_test = args
        tile_size = self.cp.tile_size
        year = orthophoto_path.name.split("-")[1]

        with rasterio.open(orthophoto_path) as ortho:
            
            window = Window(tile_x, tile_y, tile_size, tile_size)
            tile_bounds = box(*rasterio.windows.bounds(window, ortho.transform))
            
            # Require tile to be fully inside the detailed boundary
            if not boundary_ign.contains(tile_bounds):
                return
            
            # Skip if intersects test zone
            if tile_bounds.intersects(zone_test): return

            tile_data = ortho.read(window=window)
            tile_transform = rasterio.windows.transform(window, ortho.transform)

            # Apply threshold to filter out mostly black or white tiles
            greyscale_tile = np.sum(tile_data, axis=0) / 3
            
            # Black threshold.
            percentage_black_pixel = np.sum(greyscale_tile == 0) * 100 / tile_size**2
            if percentage_black_pixel > 5: 
                return
            
            # White threshold.
            percentage_white_pixel = np.sum(greyscale_tile == 255) * 100 / tile_size**2
            if percentage_white_pixel > 10: 
                return


            tile_filename = f"{session_name}_{year}_{tile_x}_{tile_y}.tif"
            tile_output_path = Path(self.pm.coarse_cropped_ortho_tif_folder, tile_filename)
            
            meta = ortho.meta.copy()
            meta.update({
                "height": tile_size, 
                "width": tile_size, 
                "transform": tile_transform
            })

            with rasterio.open(tile_output_path, "w", **meta) as dest:
                dest.write(tile_data)


            tile_geom = mapping(box(*rasterio.windows.bounds(window, ortho.transform)))
            tile_filename = f"{session_name}_{year}_{tile_x}_{tile_y}.tif"
            output_path = Path(self.pm.coarse_annotation_tif_folder, tile_filename)

            with rasterio.open(annotation_path) as annotation:
                try:
                    cropped, cropped_transform = mask(annotation, [tile_geom], crop=True)

                    resampled = np.zeros((annotation.count, tile_size, tile_size), dtype=cropped.dtype)

                    for i in range(annotation.count):
                        reproject(
                            source=cropped[i],
                            destination=resampled[i],
                            src_transform=cropped_transform,
                            src_crs=annotation.crs,
                            dst_transform=cropped_transform * annotation.transform.scale(
                                cropped.shape[2] / tile_size,
                                cropped.shape[1] / tile_size
                            ),
                            dst_crs=annotation.crs,
                            resampling=Resampling.nearest
                        )

                    out_meta = annotation.meta.copy()
                    out_meta.update({
                        "height": tile_size,
                        "width": tile_size,
                        "transform": cropped_transform * annotation.transform.scale(
                            cropped.shape[2] / tile_size,
                            cropped.shape[1] / tile_size
                        ),
                        "dtype": resampled.dtype
                    })

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(resampled)

                except Exception as e:
                    print(f"❌ Failed to process {tile_filename}: {e}")
