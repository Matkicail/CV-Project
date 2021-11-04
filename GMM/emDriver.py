import numpy as np
from numpy.linalg.linalg import _convertarray
import scipy.stats as stats
from em_centroid import *
from datasetCreator import makeMasksBinary, collectImagesAndMasks, createDataSets, getFeatureVectors
from matplotlib import pyplot as plt


#####################
#    DRIVER CODE    #
#####################

# def bayesianProb(lam, probFor, probBack):
#     """
#     Calculates the bayesian probability that the point being examined is a foreground pixel.
#     It requires a lambda (bernoulli prior) and two probabilities sampled from their respective GMMs
#     """
#     num = lam * probFor
#     den = num + (1-lam) * probBack
#     return num / den

def accuracy(predictions, mask):
    return np.sum(np.abs(predictions - mask.flatten()))/ len(mask.flatten())


def determineLam(lenBGround, lenFGround):
    """
    Find bernoulli lambda value from data set in training
    """
    return lenFGround / (lenBGround + lenFGround)


def validationAccuracy(fGroundEMCS, bGroundEMCS, validationFeatureVector, validationMasks, lam):
    validationAnswers = validationMasks.astype(int)
    # given a point see what the GMMs' probs are for it
    fGroundProbs = fGroundEMCS.samplePoint(validationFeatureVector)
    bGroundProbs = bGroundEMCS.samplePoint(validationFeatureVector)
    
    # check the prior value
    bayesianProb = lam*fGroundProbs / (lam*fGroundProbs + (1-lam)*bGroundProbs)
    flatBayes = bayesianProb.flatten()
    # find the best threshold value
    thresholds = np.linspace(start = 0, stop = 1, num=200)
    accuracies = np.array(())
    for thresh in thresholds:
        temp = np.zeros((len(flatBayes)))
        temp[np.where(flatBayes > thresh)] = 1
        temp = temp.astype(int)
        accuracies = np.append(accuracies, accuracy(temp, validationAnswers.copy()))
        print("\tCurrently on threshold {0} \r".format(thresh), end="\r")
    
    bestThreshIndex = np.argmax(accuracies)
    bestThresh = thresholds[bestThreshIndex]
    print("Best threshold is: {0}, with accuracy {1}".format(bestThresh, accuracies[bestThreshIndex]))
    return bestThresh, accuracies[bestThreshIndex]

def testingAccuracy(fGroundEMCS, bGroundEMCS, testingFeatureVector, testingMasks, lam, thresh):
    testingAnswers = testingMasks.astype(int)
    # given a point see what the GMMs' probs are for it
    fGroundProbs = fGroundEMCS.samplePoint(testingFeatureVector)
    bGroundProbs = bGroundEMCS.samplePoint(testingFeatureVector)

    # check the prior value
    bayesianProb = lam*fGroundProbs / (lam*fGroundProbs + (1-lam)*bGroundProbs)
    flatBayes = bayesianProb.flatten()
    temp = np.zeros((len(flatBayes)))
    temp[np.where(flatBayes > thresh)] = 1
    temp = temp.astype(int)
    acc = accuracy(temp, testingAnswers.copy())
    return acc

def visualRep(bestThresh, validationFeature, maskShape):
    temp = validationFeature.copy()
    fGroundProbs = fGroundEMCS.samplePoint(validationFeature[0:maskShape[1]*maskShape[2]])
    bGroundProbs = bGroundEMCS.samplePoint(validationFeature[0:maskShape[1]*maskShape[2]])
    bayesianProb = lam*fGroundProbs / (lam*fGroundProbs + (1-lam)*bGroundProbs)
    temp[:] = 0
    temp[np.where(bayesianProb > bestThresh) ] = 1 
    temp = temp.reshape(maskShape)
    firstImage = temp[0]
    plt.imshow(firstImage)
    plt.show()

# collecting data sets
print("Collecting Data sets...")
images, masks = collectImagesAndMasks()
print("Creating Data sets...")
training, validation, testing = createDataSets(images, masks)

feature=[
    ["RGB","DoG"],
    ["RGB"],
    ["RGB, DoG, HSV"],
    ["RGB"],
    ["RGB","DoG"]
]

fGroundSizes = [4,8,12,16,16]
bGroundSizes = [2,4,6,8,8]
accuracies = []
# use training to learn (the data set)
for i in range(5):
    print("Creating feature vectors...")
    bGround, fGround = getFeatureVectors(training[i][0], training[i][1], features=feature[i])
    print("Initialising EMCentroid...")

    # running the foreground EMCs
    print("Training foreground GMM")
    fGroundEMCS = EMCCentroid(fGroundSizes[i], fGround.shape[-1])
    fGroundEMCS.run(fGround, tol=1e-3)

    # running the background EMCs
    print("Training background GMM")
    bGroundEMCS = EMCCentroid(bGroundSizes[i], bGround.shape[-1])
    bGroundEMCS.run(bGround, tol=1e-3)

    print("Getting lambda values...")
    lam = determineLam(bGround.shape[0], fGround.shape[0])
    print("Evaluating model and tuning threshold for model...")
    currThresh, currAcc = validationAccuracy(fGroundEMCS, bGroundEMCS, validation[i][0], validation[i][1], lam)
    print("Training finished for iteration {0}".format(i))
    shape = (8,768,1024, fGround.shape[-1])
    # visualRep(currThresh, validation[i][0], shape)
    accuracies.append(currAcc)

accuracies = np.array(accuracies)
# np.savetxt("AccuraciesForModel.txt", accuracies)
bestModelIndex = np.argmax(accuracies)
print("Testing Process: Creating feature vectors...")
bGround, fGround = getFeatureVectors(training[bestModelIndex][0], training[bestModelIndex][1], features=feature[bestModelIndex])
print("Testing Process: Initialising EMCentroid...")

# running the foreground EMCs
print("Testing Process: Training foreground GMM")
fGroundEMCS = EMCCentroid(fGroundSizes[bestModelIndex], fGround.shape[-1])
fGroundEMCS.run(fGround, tol=1e-3)

# running the background EMCs
print("Testing Process: Training background GMM")
bGroundEMCS = EMCCentroid(bGroundSizes[bestModelIndex], bGround.shape[-1])
bGroundEMCS.run(bGround, tol=1e-3)

print("Testing Process: Getting lambda values...")
lam = determineLam(bGround.shape[0], fGround.shape[0])
print("Testing Process: Evaluating model and tuning threshold for model...")
currThresh, currAcc = testingAccuracy(fGroundEMCS, bGroundEMCS, training[0], training[1], lam)
print("Testing Process finished")
shape = (8,768,1024, fGround.shape[-1])
visualRep(currThresh, training[0], shape)
print("Current accuracy: {0}".format(currAcc))
print("Best Theta: {0}".format(currThresh))
