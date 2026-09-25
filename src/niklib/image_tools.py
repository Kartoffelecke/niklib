from pathlib import Path
from typing import List, Literal
import tifffile
import numpy as np
import skimage as sk
from readlif.reader import LifFile
import os
import warnings

def main():
    print("script ran directly")

def extract_channels(file:Path, channels: List[int]):
   return tifffile.imread(file)[channels] 

def imwrite(path:Path, data:np.ndarray, **kwargs):
    tifffile.imwrite(path, data, **kwargs)

def imsread(img_paths:Path | List[Path],) -> np.ndarray:
    """ read one or mutiple images and return a array where the image is the 1° dim"""
    if isinstance(img_paths, Path):
        img_paths = [img_paths]
    imgs = [sk.io.imread(path) for path in img_paths]
    imgs = np.stack(imgs)
    return(imgs)

def adjust_exposure(
        imgs:List[np.ndarray],
        channels:List[int],
        percentile:tuple[float,float]=(1,99),
        ):
    for ch in channels:
        low = np.percentile(imgs[:,ch,...], percentile[0])
        high = np.percentile(imgs[:,ch,...], percentile[1])
        imgs[:,ch,:,:] = sk.exposure.rescale_intensity(
            imgs[:,ch,:,:], in_range=(low,high), out_range="dtype"
        )
    return imgs
def lif_to_tif(input_file:Path, output_dir:Path, ):
    lif = LifFile(input_file)
    img_names = [x["name"] for x in lif.image_list]
    os.makedirs(output_dir, exist_ok=True)
    for img_name in img_names:
        lif_img = lif.get_image_by_name(img_name)
        cur_img = np.array(list(lif_img.get_iter_c(0, 0)))  # (C, Y, X)
        sx, sy = lif_img.scale[0], lif_img.scale[1]  # px/µm
        tifffile.imwrite(
            output_dir/f"{img_name}.tif",
            cur_img,
            imagej=True,
            resolution=(sx, sy),
            metadata={"axes": "CYX", "unit": "um"},
        )

def napari_load_tif(path:Path, viewer=None):
    """ open a tif in napari with its physical pixel size (µm) applied as scale"""
    import napari

    path = Path(path)
    with tifffile.TiffFile(path) as tif:
        data = tif.asarray()
        axes = tif.series[0].axes
        ij = tif.imagej_metadata or {}
        tags = tif.pages[0].tags
        # µm per ResolutionUnit; ImageJ tifs use NONE and store the unit in metadata
        unit_to_um = {2: 25400.0, 3: 10000.0}.get(
            tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 1, 1.0
        )

        def px_size(tag):
            if tag not in tags:
                return 1.0
            num, den = tags[tag].value
            return den / num * unit_to_um

        sizes = {"X": px_size("XResolution"), "Y": px_size("YResolution"),
                 "Z": ij.get("spacing", 1.0)}
        if unit_to_um == 1.0 and ij.get("unit") not in ("um", "micron", "µm", "\\u00B5m"):
            warnings.warn(f"{path.name}: no µm scale found, pixel sizes may be wrong")

    if viewer is None:
        viewer = napari.Viewer()
    channel_axis = axes.index("C") if "C" in axes else None
    scale = [sizes.get(a, 1.0) for a in axes if a != "C"]
    viewer.add_image(data, channel_axis=channel_axis, scale=scale, name=path.stem)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    return viewer


if __name__ == "__main__":
    main()
