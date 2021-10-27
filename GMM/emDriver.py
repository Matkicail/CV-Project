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
allLength = len(training[0]) + len(validation[0]) + len(testing[0])

# use training to learn (the data set)
print("Creating feature vectors...")
backFeats, foreFeats, flatFeat = getFeatureVectors(training[0], training[1], features=["RGB","DoG"])
print("Initialising EMCentroid...")
em = EMCCentroid(2, 4)
em.datapointResponsibilities(flatFeat[0])