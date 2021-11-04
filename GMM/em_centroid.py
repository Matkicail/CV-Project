from cv2 import meanShift
import numpy as np
from numpy.linalg.linalg import _convertarray
import scipy.stats as stats


class EMCCentroid:

    def __init__(self, numK, dataDims):
        """
        Centroid class that will be used for both background and foreground
        """
        self.numK = numK
        self.dataDims = dataDims # assuming what gets passed in is an image with all it's data points
        self.lams = np.ones(numK)/numK
        self.means = np.random.rand(numK, dataDims)
        self.covs = self.initialiseCovs(self.dataDims, numK) # set of covariances - M * D * D 

    def initialiseCovs(self, dims, numK):
        """
        Initialises the centroids 
        """
        A = np.random.rand(numK,dims,dims)
        for i in range(numK):
            A[i,:,:] += dims * np.eye(dims)
        return A

    def datapointResponsibilities(self, datapoints):
        """
        Given a set of data points - assuming stats can take in a vector and will return a vector.
        Let there be n data points, of which the dimension d is d > 1
        """
        numerators = []
        for i in range(len(self.lams)):
            numerators.append(self.lams[i] * stats.multivariate_normal.pdf(datapoints, self.means[i], self.covs[i]))
        numerators = np.array(numerators) 
        denominators = np.sum(numerators, axis = 0)
        responsibilities = numerators / denominators
        return responsibilities

    def samplePoint(self, datapoints):
        """
        Getting the probability that of a set of points which comes from this set of centroids as given by the equations in the PDF (3,4,5)
        """
        prob = 0
        for i in range(self.means.shape[0]):
            # tempProb =  stats.multivariate_normal.pdf(datapoints, self.means[i], self.covs[i])
            prob += self.lams[i] * stats.multivariate_normal.pdf(datapoints, self.means[i], self.covs[i])
        return prob

    def lambdasUpdate(self, responsibilities):
        """
        Given a set of responsibilities for current iteration calculate the new lambda values 
        responsibilities should a be a matrix of the responsibilities for each point for a given k as the row entries
        """ 
        numerators = np.sum(responsibilities, axis=1)
        denominator = np.sum(numerators)
        self.lams = numerators/denominator


        return
    
    def meanUpdates(self, datapoints, responsibilities):
        """
        Given a set of data points and responsibilities for those datapoints calculate the new means
        The responsibilities must be a matrix where each row respresents k's responsibility for each data point
        Let v be the number of data points given which have 
        """

        numerator = responsibilities @ datapoints # gets a column vector
        denominator = 1 / np.sum(responsibilities, axis = 1) # responsibility along the rows (so should be of length k and a column vector)
        self.means = (numerator.T * denominator).T # elementwise multiplication

        return

    def covariancesUpdate(self, datapoints, responsibilities):
        """
        Call this after the mean has been calculated
        Pass a set of data points and a matrix of responsibilities for those data points
        The responsibilities must be a matrix where each row respresents k's responsibility for each data point
        """
        cluster = 0
        for mean in self.means:

            # Believe this is correct but if doesnt work just go datapoints.copy() * responsibilities[cluster,:]
            tempPoints = (datapoints.copy() - mean) * responsibilities[cluster,:].reshape(datapoints.shape[0], 1)

            numerator = (tempPoints).T @ (datapoints - mean)

            denominator = np.sum(responsibilities[cluster,:])
            self.covs[cluster] = numerator / denominator 
            cluster += 1

    def EStep(self, datapoints):
        respsonsibilities = self.datapointResponsibilities(datapoints)
        return respsonsibilities

    def MStep(self, responsibilities, datapoints):
        self.lambdasUpdate(responsibilities)
        self.meanUpdates(datapoints, responsibilities)
        self.covariancesUpdate(datapoints, responsibilities)

    def randomParams(self):
        self.lams = np.ones(self.numK) / self.numK
        self.means = np.random.random((self.numK,self.dataDims))

    def run(self, datapoints, max = 1000, tol = 1e-1):

        # need conditions for convergence
        for i in range(max):
            print("\t Iteration: {0}...".format(i))
            responsibilities = self.EStep(datapoints.copy()) # at this point it is passing in the correct set
            tempLams = self.lams.copy()
            tempMeans = self.means.copy()
            tempCovs = self.covs.copy()
            self.MStep(responsibilities, datapoints)

            lamDiffs = np.linalg.norm(tempLams - self.lams)
            meanDifs = np.linalg.norm(tempMeans - self.means) / self.means.shape[0]
            covDiffs = np.linalg.norm(tempCovs - self.covs) / self.covs.shape[0]
            # or np.all(np.abs(lamDiffs - prevLamDiff) < tol
            if np.all(lamDiffs < tol): # checking for a specified degree of convergence
                print("lams were okay")
                # or np.all(np.abs(meanDifs - prevMeanDiff) < tol
                if np.all(meanDifs < tol): # checking for a specified degree of convergence
                    print("means were okay")
                    # or np.all(np.abs(covDiffs - prevCovDiff) < tol
                    if np.all(covDiffs < tol): # checking for a specified degree of convergence
                        return
                    else:
                        print("Cov diffs: {0}".format(np.abs(covDiffs)))
                else: 
                    print("Mean diffs: {0}".format(np.abs(meanDifs)))
            else: 
                print("Lam diffs: {0}".format(np.abs(lamDiffs)))
