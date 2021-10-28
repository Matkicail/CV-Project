import os
import torchvision.transforms as transforms
from natsort import natsorted
from glob import glob
from skimage import img_as_float32, img_as_ubyte
import numpy as np
from PIL import Image
from shutil import copy2, rmtree

class ReadData():
    #Give the directory and imageSet Start and End. 6 images makes 1 set, so 6*1000 = 6000 images
    def __init__(self, directory, transformImgs=None, transformMsks=None, SplitDataSet = False):
        path_pairs = None

        #Load the images from the correct directory
        if SplitDataSet:
            path_pairs = list(zip(
                natsorted(glob('../puzzle_corners_1024x768/images-1024x768/*.png')),
                natsorted(glob('../puzzle_corners_1024x768/masks-1024x768/*.png')),
            ))
        else:
            path_pairs = list(zip(
                natsorted(glob(directory + '/images-1024x768/*.png')),
                natsorted(glob(directory + '/masks-1024x768/*.png')),
            ))
        
        #Get as numpy array
        self.imgs = np.array([ipath for ipath, _ in path_pairs])
        self.msks = np.array([mpath for _, mpath in path_pairs])

        #Save transformations
        self.transformImgs = transforms.Compose(transformImgs)
        self.transformMsks = transforms.Compose(transformMsks)

        #Set up directories
        self.trainingDirectory = "./Data/Training"
        self.validationDirectory =  "./Data/Validation"
        self.testingDirectory = "./Data/Test"

        #Split the data if needed, and get the correct directories data
        if SplitDataSet:
            self.SplitData()

            path_pairs = list(zip(
                natsorted(glob(directory + '/images-1024x768/*.png')),
                natsorted(glob(directory + '/masks-1024x768/*.png')),
            ))
            self.imgs = np.array([ipath for ipath, _ in path_pairs])
            self.msks = np.array([mpath for _, mpath in path_pairs])

    def __getitem__(self, index):
        imgs = Image.open(self.imgs[index])
        msks = Image.open(self.msks[index])

        if self.transformImgs:
            imgs = self.transformImgs(imgs)
        if self.transformMsks:
            msks = self.transformMsks(msks)

        return imgs, msks

    def __len__(self):
        return len(self.imgs)

    def SplitData(self):
            #Shuffle the data
            assert len(self.imgs) == len(self.msks)
            permutations = np.random.permutation(len(self.imgs))
            self.imgs = self.imgs[permutations]
            self.msks = self.msks[permutations]

            #Delete directories if they exist
            if(os.path.isdir('./Data')):
                rmtree('./Data')

            os.makedirs(self.trainingDirectory + '/images-1024x768')
            os.makedirs(self.trainingDirectory + '/masks-1024x768')

            os.makedirs(self.validationDirectory + '/images-1024x768')
            os.makedirs(self.validationDirectory + '/masks-1024x768')

            os.makedirs(self.testingDirectory + '/images-1024x768')
            os.makedirs(self.testingDirectory + '/masks-1024x768')

            for i in range(int(len(self.imgs) * 0.70)):
                copy2(self.imgs[i], self.trainingDirectory + '/images-1024x768')
                copy2(self.msks[i], self.trainingDirectory + '/masks-1024x768')

            for i in range(int(len(self.imgs) * 0.70), int(len(self.imgs) * 0.85)):
                copy2(self.imgs[i], self.validationDirectory + '/images-1024x768')
                copy2(self.msks[i], self.validationDirectory + '/masks-1024x768')

            for i in range(int(len(self.imgs) * 0.85), int(len(self.imgs) * 1.0)):
                copy2(self.imgs[i], self.testingDirectory + '/images-1024x768')
                copy2(self.msks[i], self.testingDirectory + '/masks-1024x768')