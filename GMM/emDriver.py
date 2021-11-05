import numpy as np
from numpy.linalg.linalg import _convertarray
import scipy.stats as stats
from em_centroid import *
from datasetCreator import makeMasksBinary, collectImagesAndMasks, createDataSets, getFeatureVectors
from matplotlib import pyplot as plt
import time

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
    thresholds = np.linspace(start = 0, stop = 1, num=1000)
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
    testingMasks = np.array(testingMasks)
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
    ["RGB","DoG","HSV"]
]

fGroundSizes = [4,8,12,16,16]
bGroundSizes = [2,4,6,8,8]
accuracies = []
thresholds = []
backModels = []
frontModels = []
# use training to learn (the data set)
for i in range(5):
    print("Creating feature vectors...")
    bGround, fGround = getFeatureVectors(training[i][0], training[i][1], features=feature[i])
    print("Initialising EMCentroid...")

    # running the foreground EMCs
    print("Training foreground GMM")
    fGroundEMCS = EMCCentroid(fGroundSizes[i], fGround.shape[-1])
    fGroundEMCS.run(fGround, tol=1e-3)
    frontModels.append(fGroundEMCS)

    # running the background EMCs
    print("Training background GMM")
    bGroundEMCS = EMCCentroid(bGroundSizes[i], bGround.shape[-1])
    bGroundEMCS.run(bGround, tol=1e-3)
    backModels.append(bGroundEMCS)

    print("Getting lambda values...")
    lam = determineLam(bGround.shape[0], fGround.shape[0])
    print("Evaluating model and tuning threshold for model...")
    currThresh, currAcc = validationAccuracy(fGroundEMCS, bGroundEMCS, validation[i][0], validation[i][1], lam)
    thresholds.append(currThresh)
    print("Training finished for iteration {0}".format(i))
    shape = (8,768,1024, fGround.shape[-1])
    # visualRep(currThresh, validation[i][0], shape)
    accuracies.append(currAcc)


# ACTUAL RUNNING OF ALGORITHM

timeTraining = 00
timeInference = 0

accuracies = np.array(accuracies)
# np.savetxt("AccuraciesForModel.txt", accuracies)
bestModelIndex = np.argmax(accuracies)
numRuns = 10
for t in range(numRuns):
    learnTime = time.time()
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
    timeTraining += time.time() - learnTime 

for i in range(10):

    print("Testing Process: Evaluating model and tuning threshold for model...")
    infTime = time.time()
    currAcc = testingAccuracy(frontModels[bestModelIndex], backModels[bestModelIndex], testing[0], testing[1], lam, thresholds[bestModelIndex])
    print("Testing Process finished")
    shape = (8,768,1024, fGround.shape[-1])
    timeInference += time.time() - infTime
    visualRep(currThresh, testing[0], shape)
    print("Current accuracy: {0}".format(currAcc))
    print("Best Theta: {0}".format(currThresh))

print("Average time for training: {0}".format(timeTraining/numRuns))
print("Average time for inference: {0}".format(timeInference/numRuns))