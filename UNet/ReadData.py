import os
import torchvision.transforms as transforms
from natsort import natsorted
from glob import glob
from skimage import img_as_float32, img_as_ubyte
import numpy as np
from PIL import Image

class ReadData():
    #Give the directory and imageSet Start and End. 6 images makes 1 set, so 6*1000 = 6000 images
    def __init__(self, transform=None):
        path_pairs = list(zip(
        natsorted(glob('../puzzle_corners_1024x768/images-1024x768/*.png')),
        natsorted(glob('../puzzle_corners_1024x768/masks-1024x768/*.png')),
        ))
        self.imgs = np.array([ipath for ipath, _ in path_pairs])
        self.msks = np.array([mpath for _, mpath in path_pairs])

        self.transform = transforms.Compose(transform)

    def __getitem__(self, index):
        imgs = Image.open(self.imgs[index])
        msks = Image.open(self.msks[index])

        if self.transform:
            imgs = self.transform(imgs)
            msks = self.transform(msks)

        return imgs, msks

    def __len__(self):
        return len(self.imgs)