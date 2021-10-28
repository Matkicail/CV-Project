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

    def processData(self, datapoints):
        return np.mean(datapoints, axis = -1) # a way to transform the data point to see what the data point looks like for each channel

    def initialiseCovs(self, dims, numK, eps=1e-6):
        A = np.random.rand(numK,dims,dims)
        for i in range(numK):
            A[i,:,:] += eps + np.eye(dims)
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

    def lambdasUpdate(self, responsibilities):
        """
        Given a set of responsibilities for current iteration calculate the new lambda values 
        responsibilities should a be a matrix of the responsibilities for each point for a given k as the row entries
        """ 
        numerators = np.sum(responsibilities, axis=1)
        denominator = np.sum(numerators)
        self.lams = numerators/denominator
        a = 2 # debug step
        return
    
    def meanUpdates(self, datapoints, responsibilities):
        """
        Given a set of data points and responsibilities for those datapoints calculate the new means
        The responsibilities must be a matrix where each row respresents k's responsibility for each data point
        Let v be the number of data points given which have 
        """
        a = 2
        numerator = responsibilities @ datapoints # gets a k*n @ n*v => result is k*v column vector
        denominator = 1 / np.sum(responsibilities, axis = 1) # responsibility along the rows (so should be of length k and a column vector)
        self.means = (numerator.T * denominator).T # elementwise multiplication
        a = 2 # debug step
        return

    def covariancesUpdate(self, datapoints, responsibilities):
        """
        Call this after the mean has been calculated
        Pass a set of data points and a matrix of responsibilities for those data points
        The responsibilities must be a matrix where each row respresents k's responsibility for each data point
        """
        cluster = 0
        steps = int(len(datapoints)/ 100)
        for mean in self.means:
            numerator = 0
            cnt = 0
            perc = 0

            tempPoints = (datapoints.copy() - mean) * responsibilities[cluster,:].reshape(datapoints.shape[0], 1)
            numerator = (tempPoints).T @ (datapoints - mean)
            # for datapoint in datapoints:
            #     temp = datapoint.copy()
            #     temp = temp.reshape((len(temp),1))
            #     numerator += responsibilities[cluster][cnt]* ((temp - mean)@((temp-mean).T))
            #     if cnt % steps == 0:
            #         print('\t\t\tUpdate %d percent' % perc, end='\r')
            #         perc += 1
            #     cnt += 1
            denominator = np.sum(responsibilities[cluster,:])
            self.covs[cluster] = numerator / denominator 
            a = 2 # debug step
    def EStep(self, datapoints):
        # respsonsibilities = []
        # for datapoint in datapoints:
        #     respsonsibilities.append(self.datapointResponsibilities(datapoint))
        respsonsibilities = self.datapointResponsibilities(datapoints)
        a = 2 # just as a debug step
        return respsonsibilities

    def MStep(self, responsibilities, datapoints):
        self.lambdasUpdate(responsibilities)
        self.meanUpdates(datapoints, responsibilities)
        self.covariancesUpdate(datapoints, responsibilities)

    def randomParams(self):
        self.lams = np.ones(self.numK) / self.numK
        self.means = np.random.random((self.numK,self.dataDims))

    def run(self, datapoints, max = 1000, tol = 1e-2):

        # need conditions for convergence

        # current debug to check process is working
        for i in range(max):
            print("\t Iteration: {0}...".format(i))
            responsibilities = self.EStep(datapoints.copy()) # at this point it is passing in the correct set
            tempLams = self.lams.copy()
            tempMeans = self.means.copy()
            tempCovs = self.covs.copy()
            self.MStep(responsibilities, datapoints)
            if np.all(tempLams - self.lams < tol):
                print("lams were okay")
                if np.all(tempMeans - self.means < tol):
                    print("means were okay")
                    if np.all(tempCovs - self.covs < tol):
                        print("covs were okay")
                        print("Finished Training...")
                        return