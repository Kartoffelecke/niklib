import tifffile
import numpy as np
from niklib import image_tools

def test_extract_channel(tmp_path):
    img = np.random.rand(5,10,10)
    path = tmp_path / "random_img.tif"
    tifffile.imwrite(path, img)
    assert image_tools.extract_channels(path, [0,2]).shape == (2,10,10)
