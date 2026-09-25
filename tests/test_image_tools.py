import tifffile
import numpy as np
from niklib import image_tools

def test_extract_channel(tmp_path):
    img = np.random.rand(5,10,10)
    path = tmp_path / "random_img.tif"
    tifffile.imwrite(path, img)
    assert image_tools.extract_channels(path, [0,2]).shape == (2,10,10)

def test_adjust_exposure_one_channel():
    imgs = np.random.rand(5, 3, 10, 10)
    result = image_tools.adjust_exposure(imgs, [0])
    assert result.shape == imgs.shape
