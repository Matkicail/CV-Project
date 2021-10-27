import numpy as np
from numpy.linalg.linalg import _convertarray
import scipy.stats as stats


class EMCCentroid:

    def __init__(self, numK, dataDims):
        self.numK = numK
        self.dataDims = dataDims # assuming what gets passed in is an image with all it's data points
        self.lams = np.ones(numK)/numK
        self.means = np.random.rand(numK, dataDims)
        self.covs = self.initialiseCovs(self.dataDims, numK) # set of covariances - M * D * D 

    def initialiseCovs(self, dims, numK):
        A = np.random.rand(numK,dims,dims)
        for i in range(numK):
            A[i,:,:] += A[i,:,:].T
        return A

    def datapointResponsibilities(self, datapoints):
        """
        Given a set of data points - assuming stats can take in a vector and will return a vector.
        Let there be n data points, of which the dimension d is d > 1
        """
        numerators = self.lams * stats.multivariate_normal.pdf(datapoints, self.means, self.covs)
        denominators = np.sum(numerators)
        responsibilities = numerators / denominators
        return responsibilities

    def lambdasUpdate(self, responsibilities):
        """
        Given a set of responsibilities for current iteration calculate the new lambda values 
        responsibilities should a be a matrix of the responsibilities for each point for a given k as the row entries
        """ 
        numerators = np.sum(responsibilities, axis=0)
        denominator = np.sum(numerators)
        self.lams = numerators/denominator
        return
    
    def meanUpdates(self, datapoints, responsibilities):
        """
        Given a set of data points and responsibilities for those datapoints calculate the new means
        The responsibilities must be a matrix where each row respresents k's responsibility for each data point
        Let v be the number of data points given which have 
        """
        numerator = responsibilities @ datapoints.T # gets a k*n @ n*v => result is k*v column vector
        denominator = 1 / np.sum(responsibilities, axis = 0) # responsibility along the rows (so should be of length k and a column vector)
        self.means = numerator * denominator # elementwise multiplication
        return

    def covariancesUpdate(self, datapoints, responsibilities):
        """
        Call this after the mean has been calculated
        Pass a set of data points and a matrix of responsibilities for those data points
        The responsibilities must be a matrix where each row respresents k's responsibility for each data point
        """
        internal = datapoints - self.means # n*v vector
        internal = internal @ internal.T # returns an n*n matrix 
        numerators = np.sum(responsibilities @ internal, axis = 0) # k*n @ n*n => k*n, so need to sum along the columns
        denominator = np.sum(responsibilities, axis = 0) # getting the total responsibility for K along a row
        self.covs = numerators / denominator

    def EStep(self, datapoints):
        return self.datapointResponsibilities(datapoints)

    def MStep(self, responsibilities, datapoints):
        self.lambdasUpdate(responsibilities)
        self.meanUpdates(datapoints, responsibilities)
        self.covariancesUpdate(datapoints, responsibilities)

    def randomParams(self):
        self.lams = np.ones(self.numK) / self.numK
        self.means = np.random.random((self.numK,self.dataDims))
