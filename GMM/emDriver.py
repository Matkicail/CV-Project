import numpy as np
from numpy.linalg.linalg import _convertarray
import scipy.stats as stats
from em_centroid import EMCCentroid
from datasetCreator import makeMasksBinary, collectImagesAndMasks, createDataSets, getFeatureVectors



#####################
#    DRIVER CODE    #
#####################

# collecting data sets
print("Collecting Data sets...")
images, masks = collectImagesAndMasks()
print("Creating Data sets...")
training, validation, testing = createDataSets(images, masks)

# use training to learn (the data set)
print("Creating feature vectors...")
bGround, fGround = getFeatureVectors(training[0], training[1], features=["RGB","DoG"])
print("Initialising EMCentroid...")

# running the foreground EMCs
print("Training foreground GMM")
fGroundEMCS = EMCCentroid(2, fGround.shape[-1])
fGroundEMCS.run(fGround)

# running the background EMCs
print("Training background GMM")
bGroundEMCS = EMCCentroid(2, fGround.shape[-1])
bGroundEMCS.run(bGround)

print("Training finished")
