from pathlib import Path
from typing import List, Literal
import tifffile
import numpy as np
import skimage as sk
from readlif.reader import LifFile
import os
import re
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
LUT_RGB = {
    "red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1), "cyan": (0, 1, 1),
    "magenta": (1, 0, 1), "yellow": (1, 1, 0), "gray": (1, 1, 1), "grey": (1, 1, 1),
}

def _lif_channel_info(element, n_ch:int):
    """ channel names (dye names, confocal only) and LUT color names of a .lif image
    element, None where missing"""
    colors = [
        c.get("LUTName", "").lower() or None for c in
        element.findall("./Data/Image/ImageDescription/Channels/ChannelDescription")
    ]
    # sequential scans: every sequence adds its active detectors as channels, in order
    attachment = "./Data/Image/Attachment/"
    settings = (
        element.findall(attachment + "LDM_Block_Sequential/LDM_Block_Sequential_List/ATLConfocalSettingDefinition")
        or element.findall(attachment + "ATLConfocalSettingDefinition")
    )
    names = []
    for setting in settings:
        dyes = {b.get("Channel"): b.get("DyeName", "") for b in setting.findall("./Spectro/MultiBand")}
        for det in setting.findall("./DetectorList/Detector"):
            if det.get("IsActive") == "1":
                dye = dyes.get(det.get("Channel"), "").split("/")[-1]  # "Leica/DAPI" -> "DAPI"
                names.append(dye or det.get("Name"))
    if len(names) != n_ch:
        names = [None] * n_ch
    if len(colors) != n_ch:
        colors = [None] * n_ch
    return names, colors

def _lut(color:str | None) -> np.ndarray:
    """ ImageJ LUT (3, 256) uint8 ramping from black to `color`"""
    rgb = LUT_RGB.get(color, (1, 1, 1))
    return (np.outer(rgb, np.arange(256))).astype(np.uint8)

def lif_to_tif(input_file:Path, output_dir:Path, ):
    lif = LifFile(input_file)
    # readlif doesn't expose channel names/colors, get them from the xml header
    elements = [e for e in lif.xml_root.iter("Element") if e.find("./Data/Image") is not None]
    if [e.get("Name") for e in elements] != [x["name"].split("/")[-1] for x in lif.image_list]:
        warnings.warn(f"{input_file}: could not match images to xml, skipping channel names/colors")
        elements = [None] * len(lif.image_list)
    os.makedirs(output_dir, exist_ok=True)
    for i, element in enumerate(elements):
        lif_img = lif.get_image(i)
        cur_img = np.array(list(lif_img.get_iter_c(0, 0)))  # (C, Y, X)
        sx, sy = lif_img.scale[0], lif_img.scale[1]  # px/µm
        metadata = {"axes": "CYX", "unit": "um", "mode": "composite"}
        if element is not None:
            names, colors = _lif_channel_info(element, cur_img.shape[0])
            if all(names):
                metadata["Labels"] = names
            if any(colors):
                metadata["LUTs"] = [_lut(c) for c in colors]
        tifffile.imwrite(
            output_dir/f"{lif_img.name}.tif",
            cur_img,
            imagej=True,
            resolution=(sx, sy),
            metadata=metadata,
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

def _selected_rectangle(viewer, required:bool = True):
    """ world coordinates of the corners of the selected shape in the active (or topmost)
    shapes layer. a layer with a single shape needs no selection. without any shape,
    None is returned if not `required`"""
    from napari.layers import Shapes

    shapes = viewer.layers.selection.active
    if not isinstance(shapes, Shapes):
        shapes = next((l for l in reversed(viewer.layers) if isinstance(l, Shapes)), None)
    if shapes is None or len(shapes.data) == 0:
        if required:
            raise ValueError("draw a rectangle in a shapes layer first")
        return None
    selected = list(shapes.selected_data) or ([0] if len(shapes.data) == 1 else [])
    if len(selected) != 1:
        raise ValueError(f"select exactly one rectangle in '{shapes.name}'")
    return [np.asarray(shapes.data_to_world(v)) for v in shapes.data[selected[0]]]

def _rectangle_slices(layer, corners):
    """ Y and X slices of `layer` covered by the bounding box of `corners` (world
    coordinates), None if they don't overlap"""
    yx = np.array([
        layer.world_to_data(np.pad(c, (max(layer.ndim - len(c), 0), 0)))[-2:]
        for c in corners
    ])
    shape = np.array(layer.data.shape[-3:-1] if layer.rgb else layer.data.shape[-2:])
    start = np.clip(np.floor(yx.min(0)).astype(int), 0, shape)
    stop = np.clip(np.ceil(yx.max(0)).astype(int), 0, shape)
    if np.any(stop <= start):
        return None
    return slice(start[0], stop[0]), slice(start[1], stop[1])

def napari_crop(viewer, layers=None):
    """ crop image layers (default: all, i.e. every channel) in place to the selected
    rectangle of the active shapes layer (or the topmost one). the cropped image
    keeps its position, rotated shapes are cropped to their bounding box"""
    from napari.layers import Image

    corners = _selected_rectangle(viewer)
    if layers is None:
        layers = [l for l in viewer.layers if isinstance(l, Image)]
    for layer in layers:
        crop = _rectangle_slices(layer, corners)
        if crop is None:
            warnings.warn(f"rectangle does not overlap '{layer.name}', not cropped")
            continue
        start = np.array([crop[0].start, crop[1].start])
        if layer.rgb:
            crop = crop + (slice(None),)
        translate = np.array(layer.translate, dtype=float)
        translate[-2:] += start * np.asarray(layer.scale)[-2:]
        layer.data = layer.data[(Ellipsis, *crop)]
        layer.translate = translate
    return layers

def _current_plane(viewer, layer) -> np.ndarray:
    """ the 2D (Y, X) plane of `layer` currently shown in the viewer"""
    data = layer.data
    if layer.ndim > 2:
        point = np.asarray(viewer.dims.point)[-layer.ndim:]
        idx = np.round(layer.world_to_data(point)[:-2]).astype(int)
        idx = np.clip(idx, 0, np.array(data.shape[:layer.ndim - 2]) - 1)
        data = data[tuple(idx)]
    return np.asarray(data)

def _render(plane:np.ndarray, layer, cmap) -> np.ndarray:
    """ float RGB (Y, X, 3) of `plane` with the layer's contrast limits and gamma.
    cmap is a napari Colormap or "black" for inverted gray"""
    lo, hi = layer.contrast_limits
    v = np.clip((plane.astype(float) - lo) / max(hi - lo, 1e-12), 0, 1) ** layer.gamma
    if isinstance(cmap, str) and cmap == "black":
        return np.repeat((1 - v)[..., None], 3, axis=-1)
    return cmap.map(v.ravel())[:, :3].reshape(*v.shape, 3)

def _burn_scalebar(rgb:np.ndarray, bar_px:int) -> np.ndarray:
    """ black bar on a white box in the bottom-right corner, no text"""
    h, w = rgb.shape[:2]
    t = max(3, round(0.012 * min(h, w)))  # bar thickness, also padding
    box_h, box_w = 3 * t, bar_px + 2 * t
    if box_h > h or box_w > w:
        raise ValueError(f"scale bar ({bar_px} px) does not fit into the {w} px wide image")
    out = rgb.copy()
    out[h - box_h:, w - box_w:] = 1
    out[h - 2 * t:h - t, w - t - bar_px:w - t] = 0
    return out

def napari_make_figure(
        viewer,
        outdir:Path,
        prefix:str = "figure",
        channels:List[int | str] | None = None,
        single_cmap:str | None = "black",
        overlay:List[int | str] | None = None,
        scalebar:Literal["extra", "overlay"] | None = "extra",
        scalebar_len:float = 20,
        ) -> List[Path]:
    """ export figure panels as 8-bit RGB tifs from the selected rectangle (if any) or
    the whole image, using the current contrast limits and gamma of each layer.

    channels: image layers (index or name, default all) exported as single images
    single_cmap: colormap of all single images, "black" = inverted gray,
        None = each channel's own colormap
    overlay: channels blended additively in their own colormaps (default all, [] = none)
    scalebar: "extra" = additional copy with scale bar for every image, "overlay" = only
        for the overlay, None = no scale bars
    scalebar_len: scale bar length in the unit of the image (µm)
    """
    from napari.layers import Image
    from napari.utils.colormaps import ensure_colormap

    image_layers = [l for l in viewer.layers if isinstance(l, Image)]
    names = [l.name for l in image_layers]

    def pick(keys):
        if keys is None:
            return image_layers
        picked = []
        for k in keys:
            if isinstance(k, str) and k in names:
                picked.append(image_layers[names.index(k)])
            elif isinstance(k, int) and -len(names) <= k < len(names):
                picked.append(image_layers[k])
            else:
                raise ValueError(f"unknown channel {k!r}, available: {names} (or their index)")
        return picked

    single_layers, overlay_layers = pick(channels), pick(overlay)
    for layer in set(single_layers + overlay_layers):
        if layer.rgb:
            raise ValueError(f"'{layer.name}' is an RGB layer, only single channels are supported")
    if single_cmap is None or single_cmap == "black":
        cmap = single_cmap
    else:
        cmap = ensure_colormap(_to_colormap(single_cmap))

    corners = _selected_rectangle(viewer, required=False)

    def region(layer):
        plane = _current_plane(viewer, layer)
        if corners is None:
            return plane
        crop = _rectangle_slices(layer, corners)
        if crop is None:
            raise ValueError(f"rectangle does not overlap '{layer.name}'")
        return plane[crop]

    outdir = Path(outdir)
    os.makedirs(outdir, exist_ok=True)
    written = []

    def save(rgb, name, px, with_scalebar):
        variants = [(name, rgb)]
        if with_scalebar:
            variants.append((f"{name}_scalebar", _burn_scalebar(rgb, round(scalebar_len / px))))
        for n, img in variants:
            safe = re.sub(r"[\s/\\]+", "_", n)  # "ALEXA 647" -> "ALEXA_647"
            path = outdir / f"{prefix}_{safe}.tif"
            tifffile.imwrite(
                path,
                (np.clip(img, 0, 1) * 255).round().astype(np.uint8),
                imagej=True,
                resolution=(1 / px, 1 / px),
                metadata={"unit": "um"},
            )
            written.append(path)

    for layer in single_layers:
        rgb = _render(region(layer), layer, layer.colormap if cmap is None else cmap)
        save(rgb, layer.name, layer.scale[-1], scalebar == "extra")

    if overlay_layers:
        planes = [region(l) for l in overlay_layers]
        if len({p.shape for p in planes}) > 1:
            raise ValueError("overlay channels have different sizes: "
                             + ", ".join(f"{l.name} {p.shape}" for l, p in zip(overlay_layers, planes)))
        rgb = sum(_render(p, l, l.colormap) for p, l in zip(planes, overlay_layers))
        save(rgb, "overlay", overlay_layers[0].scale[-1], scalebar in ("extra", "overlay"))
    return written


if __name__ == "__main__":
    main()
