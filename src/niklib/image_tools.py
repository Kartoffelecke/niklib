from pathlib import Path
from typing import List
import tifffile
import numpy as np

def main():
    print("script ran directly")

def extract_channels(file:Path, channels: List[int]):
   return tifffile.imread(file)[channels] 

def imwrite(path:Path, data:np.ndarray, **kwargs):
    tifffile.imwrite(path, data, **kwargs)

if __name__ == "__main__":
    main()
