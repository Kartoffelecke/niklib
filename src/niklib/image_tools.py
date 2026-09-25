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
        cur_img = np.array(
            list(lif.get_image_by_name(img_name).get_iter_c(0, 0))
        )
        tifffile.imwrite(output_dir/f"{img_name}.tif",cur_img,imagej=True)


if __name__ == "__main__":
    main()
