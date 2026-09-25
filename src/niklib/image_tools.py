from pathlib import Path
from typing import List, Literal
import tifffile
import numpy as np
import skimage as sk
from readlif.reader import LifFile
import os
import warnings
import xml.etree.ElementTree as ET

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

def _tif_channel_info(tif:tifffile.TiffFile, n_ch:int):
    """ channel names and colors stored in an OME or ImageJ tif, None where missing"""
    names, colors = [None] * n_ch, [None] * n_ch
    if tif.ome_metadata:
        pixels = ET.fromstring(tif.ome_metadata).find("{*}Image/{*}Pixels")
        channels = pixels.findall("{*}Channel") if pixels is not None else []
        for i, ch in enumerate(channels[:n_ch]):
            names[i] = ch.get("Name")
            if ch.get("Color") is not None:
                rgba = int(ch.get("Color")) & 0xFFFFFFFF  # signed int32 RGBA
                colors[i] = tuple(((rgba >> s) & 255) / 255 for s in (24, 16, 8))
    ij = tif.imagej_metadata or {}
    # tifffile returns a bare str / (3, 256) array instead of a list for single entries
    labels = ij.get("Labels", [])  # one per plane, channels vary fastest
    labels = [labels] if isinstance(labels, str) else list(labels)
    luts = ij.get("LUTs", [])  # (3, 256) uint8 per channel
    luts = [luts] if isinstance(luts, np.ndarray) and luts.ndim == 2 else list(luts)
    for i in range(n_ch):
        if names[i] is None and i < len(labels) and labels[i]:
            names[i] = labels[i]
        if colors[i] is None and i < len(luts):
            colors[i] = luts[i]
    return names, colors

def _to_colormap(color):
    """ napari colormap name, color name/hex, rgb tuple or ImageJ LUT -> napari colormap"""
    from napari.utils.colormaps import AVAILABLE_COLORMAPS, Colormap
    from napari.utils.colormaps.standardize_color import transform_color

    if isinstance(color, str) and color in AVAILABLE_COLORMAPS:
        return color
    if isinstance(color, np.ndarray):
        rgba = np.c_[color.T / 255, np.ones(color.shape[1])]
    else:
        rgba = np.stack([[0, 0, 0, 1], transform_color(color)[0]])
    r, g, b = (rgba[-1, :3] * 255).round().astype(int)
    return Colormap(rgba, name=f"#{r:02x}{g:02x}{b:02x}")

def napari_load_tif(
        path:Path,
        viewer=None,
        names:List[str] | None = None,
        colors:List[str] | None = None,
        ):
    """ open a tif in napari with its physical pixel size (µm) applied as scale.
    channel names and colors are taken from the tif if available, `names` and
    `colors` (napari colormap names like "green" or any color, e.g. "#ff8800")
    override them"""
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

        channel_axis = axes.index("C") if "C" in axes else None
        n_ch = data.shape[channel_axis] if channel_axis is not None else 1
        tif_names, tif_colors = _tif_channel_info(tif, n_ch)

    for arg, given in (("names", names), ("colors", colors)):
        if given is not None and len(given) != n_ch:
            raise ValueError(f"{path.name} has {n_ch} channels but {len(given)} {arg} were given")
    names = names or [n or f"{path.stem} [{i}]" for i, n in enumerate(tif_names)]
    colors = colors or tif_colors
    kwargs = {}
    if any(c is not None for c in colors):
        kwargs["colormap"] = [_to_colormap(c if c is not None else "gray") for c in colors]
    if channel_axis is None:
        names = names[0]
        kwargs = {k: v[0] for k, v in kwargs.items()}

    if viewer is None:
        viewer = napari.Viewer()
    scale = [sizes.get(a, 1.0) for a in axes if a != "C"]
    viewer.add_image(data, channel_axis=channel_axis, scale=scale, name=names, **kwargs)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    return viewer


if __name__ == "__main__":
    main()
