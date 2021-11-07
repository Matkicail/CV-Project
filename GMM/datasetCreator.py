from numpy.core.fromnumeric import size
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

def create5KFolds(sizeOfData): # update this
    """
    Given a set of indices create our 5-K folds
    """
    # do not leak validation into testing
    # do not leak testing into validation
    # dont use validation data that was used to train
    
    # first segment testing indices away from the data set
    np.random.seed(0) # seed after each action due to how np.random works (readjusts seed each time)
    temp = np.arange(start = 0 , stop = sizeOfData)
    temp = np.random.permutation(temp)
    testingIndices = temp[0:8]
    
    # possible to permute 
    remainingSet = temp[8:]
    possibleVals = []
    trainingSet = []

    for i in range(5):
        tempVals = []
        tempTrain = []
        for j in range(40):
            if j < 8:
                tempVals.append(remainingSet[(i*8+j)%40])
            else:
                tempTrain.append(remainingSet[(i*8+j)%40])
        possibleVals.append(tempVals)   
        trainingSet.append(tempTrain)

    return testingIndices, possibleVals, trainingSet

# generate entries to be used for training, validation and testing
def createDataSets(images, masks):

    """
    Creates a custom data set of training, validation and testing sets for a set of images and their respective masks
    Returns training set, validation set, testing set.
    This returns the 5-fold cross validation set of data.
    """
    testingIndices, validationFolds, trainingFolds = create5KFolds(len(images))
    
    testing = [images[testingIndices], masks[testingIndices]]
    validation = [
        [images[validationFolds[0]], masks[validationFolds[0]]],
        [images[validationFolds[1]], masks[validationFolds[1]]],
        [images[validationFolds[2]], masks[validationFolds[2]]],
        [images[validationFolds[3]], masks[validationFolds[3]]],
        [images[validationFolds[4]], masks[validationFolds[4]]]
    ]
    training = [
        [images[trainingFolds[0]], masks[trainingFolds[0]]],
        [images[trainingFolds[1]], masks[trainingFolds[1]]],
        [images[trainingFolds[2]], masks[trainingFolds[2]]],
        [images[trainingFolds[3]], masks[trainingFolds[3]]],
        [images[trainingFolds[4]], masks[trainingFolds[4]]]
    ]
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
    if "HSV" in features: # HSV
        dataFeatures = np.concatenate( (dataFeatures , color.rgb2hsv(images)), axis=3)

    # collect the features for the background and the foregroud

    bGround = images[np.where(masks == 1)]
    fGround = images[np.where(masks == 0)]

    return bGround, fGround