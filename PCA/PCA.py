import numpy as np

class PCA:
    def __init__(self, X):
        self.X = X

        self.R = np.corrcoef(self.X, rowvar=False)

        avgs = np.mean(self.X, axis=0)
        stds = np.std(self.X, axis=0)
        self.Xstd = (self.X - avgs) / stds

        self.Cov = np.cov(self.Xstd, rowvar=False)

        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.Cov)

        k_des = [k for k in reversed(np.argsort(self.eigenvalues))]
        self.alpha = self.eigenvalues[k_des]
        self.a = self.eigenvectors[:, k_des]

        for j in range(len(self.alpha)):
            minim = np.min(self.a[:, j])
            maxim = np.max(self.a[:, j])
            if np.abs(minim) > np.abs(maxim):
                self.a[:, j] = -self.a[:, j]

        self.C = self.Xstd @ self.a

        self.Rxc = self.a * np.sqrt(self.alpha)

        self.C2 = self.C * self.C

    def getCorr(self):
        return self.R

    def getXstd(self):
        return self.Xstd

    def getCov(self):
        return self.Cov

    def getEigenvalues(self):
        return self.alpha

    def getEigenvectors(self):
        return self.a

    def getFactorLoadings(self):
        return self.Rxc

    def getComponents(self):
        return self.C

    def getScores(self):
        return self.C / np.sqrt(self.alpha)

    def getObsQuality(self):
        C2Sum = np.sum(self.C2, axis=1)
        return np.transpose(np.transpose(self.C2) / C2Sum)

    def getBeta(self):
        return self.C2 / (self.alpha * self.X.shape[0])

    def getCommun(self):
        R2 = np.square(self.Rxc)
        return np.cumsum(R2, axis=1)
