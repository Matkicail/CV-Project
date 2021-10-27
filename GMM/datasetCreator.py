import skimage
from skimage import io as imageio
from skimage import filters, color
import numpy as np
import cv2
from natsort import natsorted
import glob

from skimage import data

def makeMasksBinary(masks):
    for mask in masks:
        mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
    return masks

# collect all the images
def collectImagesAndMasks():
    """
    Function to collect images and masks from the data set (note the folder must be in your curr directory at current)
    Returns: images and masks
    """
    path_pairs = list(zip(
        natsorted(glob.glob('./puzzle_corners_1024x768/images-1024x768/*.png')),
        natsorted(glob.glob('./puzzle_corners_1024x768/masks-1024x768/*.png')),
        )
    )
    images = np.array([skimage.img_as_float(imageio.imread(ipath)) for ipath, _ in path_pairs])
    masks = np.array([skimage.img_as_float(imageio.imread(mpath)) for _, mpath in path_pairs])
    masks = makeMasksBinary(masks)
    return images, masks

# generate entries to be used for training, validation and testing
def createDataSets(images, masks):

    """
    Creates a custom data set of training, validation and testing sets for a set of images and their respective masks
    Returns training set, validation set, testing set
    """

    indices = np.arange(start=0, stop = len(images))
    indices = np.random.permutation(indices)
    trainingIndices = indices[0:int(len(indices)*0.70)] # first 70%
    validationIndices = indices[int(len(indices)*0.70):int(len(indices)*0.85)] # second last set of 15%
    testingIndices = indices[int(len(indices)*0.85):] # last 15%
    
    trainingImages = images[trainingIndices]
    trainingMasks = masks[trainingIndices]
    training = [trainingImages, trainingMasks]
    validationImages = images[validationIndices]
    validationMasks = images[validationIndices]
    validation = [validationImages, validationMasks]
    testingImages = images[testingIndices]
    testingMasks = masks[testingIndices]
    testing = [testingImages, testingMasks]
    return training, validation, testing

def getFeatureVectors(images, masks, features, sigmaSmall = 3, sigmaLarge = 6):
    """
    Function takes in images, masks, a set of features with text for each feature wanted and optionally a setting for the small and large sigma values for DoG.
    It returns a set of features for the background and foreground from the data set given.
    """
    # need the RGB from all of them so this will be standard
    dataFeatures = np.array(())
    if "RGB" in features:
        dataFeatures = images
    if "DoG" in features:
        guassSmall = filters.gaussian(color.rgb2gray(images), sigmaSmall)
        guassLarge = filters.gaussian(color.rgb2gray(images), sigmaLarge)
        dog = guassLarge - guassSmall
        dog = dog.reshape((dog.shape[0], dog.shape[1], dog.shape[2], 1))
        dataFeatures = np.concatenate((dataFeatures, dog), axis=3)
    if "OtherFeature" in features:
        raise NotImplementedError
    background = []
    foreground = []
    count = 0
    for mask in masks:
        backIndices = np.where(mask == 0)
        foreIndices = np.where(mask == 1)
        background.append(images[count][backIndices])
        foreground.append(images[count][foreIndices])
    
    # get initial estimates for forground and background
    flattenedFeatures = []
    for dataFeature in dataFeatures:
        flattenedFeatures.append(dataFeature.flatten())

    return background, foreground, flattenedFeatures