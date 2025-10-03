import tifffile
import numpy as np
from tqdm import tqdm
from PIL import Image
import geopandas as gpd
from pathlib import Path

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from rasterio.transform import from_origin
from rasterio.windows import from_bounds, Window


def convert_one_tiff_to_png(args: tuple[Path, Path]) -> None:
    """Convert an image or annotation from TIFF to PNG without using GDAL."""
    filepath, output_dir = args
    png_output_path = Path(output_dir, f'{filepath.stem}.png')

    raster_data = tifffile.imread(filepath)

    if raster_data.ndim == 3:
        # TIFFs are typically in (height, width, channels) for tifffile
        if raster_data.shape[0] in (3, 4) and raster_data.shape[0] < raster_data.shape[2]:
            # Fix for (bands, H, W) layout
            raster_data = np.transpose(raster_data[:3], (1, 2, 0))

        image = Image.fromarray(raster_data[..., :3].astype(np.uint8), mode="RGB")

    elif raster_data.ndim == 2:
        image = Image.fromarray(raster_data.astype(np.uint8), mode="L")

    else:
        raise ValueError(f"Unexpected image format: {raster_data.shape}")

    image.save(png_output_path)


def incremental_merge_tifs_windowed(tif_files: list[Path], output_path: Path, tile_size:int=512):

    # Reference metadata
    with rasterio.open(tif_files[0]) as ref:
        res = ref.res
        crs = ref.crs
        dtype = ref.dtypes[0]

    # Global extent
    bs = [rasterio.open(t).bounds for t in tif_files]
    minx = min(b.left for b in bs); miny = min(b.bottom for b in bs)
    maxx = max(b.right for b in bs); maxy = max(b.top for b in bs)

    width = int(np.ceil((maxx - minx)/res[0]))
    height = int(np.ceil((maxy - miny)/res[1]))
    transform = from_origin(minx, maxy, *res)

    meta = {
        "driver":"GTiff",
        "height":height,
        "width":width,
        "count":1,
        "dtype":dtype,
        "crs":crs,
        "transform":transform,
        "compress":"LZW",
        "tiled":True,
        "blockxsize":tile_size,
        "blockysize":tile_size,
        "nodata":0
    }

    # Create empty output
    with rasterio.open(output_path, "w", **meta) as dst:
        pass

    with rasterio.open(output_path, "r+") as dst:
        for tif in tqdm(tif_files, desc="Merging + Masking tiles"):
            with rasterio.open(tif) as src:
                src_bounds = src.bounds
                win_overall = from_bounds(*src_bounds, transform=transform).round_offsets().round_lengths()

                for i in range(win_overall.row_off, win_overall.row_off + win_overall.height, tile_size):
                    for j in range(win_overall.col_off, win_overall.col_off + win_overall.width, tile_size):
                        h = min(tile_size, win_overall.row_off + win_overall.height - i)
                        w = min(tile_size, win_overall.col_off + win_overall.width - j)
                        dst_win = Window(j, i, w, h)
                        dst_bounds = dst.window_bounds(dst_win)

                        # Get matching window from src
                        src_win = from_bounds(*dst_bounds, transform=src.transform).round_offsets().round_lengths()
                        if src_win.width <= 0 or src_win.height <= 0:
                            continue

                        data = src.read(1, window=src_win, out_shape=(int(src_win.height), int(src_win.width)))
                        dst_data = dst.read(1, window=dst_win)

                        valid = (data > 0) & (dst_data == 0)
                        if valid.any():
                            dst_data[valid] = data[valid]
                            dst.write(dst_data, 1, window=dst_win)

    print(f"✅ Lagoon-masked merged raster saved to: {output_path}")


# Function to clip raster to polygons
def clip_raster_to_polygons(input_raster: Path, output_raster: Path, polygons_gdfs: list[gpd.GeoDataFrame]):
    if not input_raster.exists():
        print(f"File not found: {input_raster.name}")
        return

    with rasterio.open(input_raster) as src:
        raster_crs = src.crs
        print(f"Processing {input_raster.name} | CRS: {raster_crs}")

        # Reproject lagoon polygons to raster CRS
        polygons = []
        for gdf in polygons_gdfs:
            polygons += [
                transform_geom(gdf.crs.to_string(), raster_crs.to_string(), geom)
                for geom in gdf.geometry
            ]

        # Clip raster
        out_image, out_transform = mask(src, polygons, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "DEFLATE",
            "tiled": True,
        })

        if out_meta.get("nodata") is None:
            out_meta["nodata"] = 0

    with rasterio.open(output_raster, "w", **out_meta) as dst:
        dst.write(out_image)

    print(f"Saved clipped raster: {output_raster}")