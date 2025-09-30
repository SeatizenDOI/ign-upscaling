import tifffile
import numpy as np
from PIL import Image
from pathlib import Path

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