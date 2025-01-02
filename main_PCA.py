import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import graphics as graphics


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


# extract the data from the .csv file in a pandas.DataFrame
table = pd.read_csv("./dataIN/cards.csv")
# print(table)

obs = table.index[:]
variables = table.columns[:]
X = table[variables].values
numberObsVariables = X.shape[1]

pcaModel = PCA(X)
correlation = pcaModel.getCorr()
eigenvalues = pcaModel.getEigenvalues()
eigenvectors = pcaModel.getEigenvectors()
factorLoadings = pcaModel.getFactorLoadings()
components = pcaModel.getComponents()
scores = pcaModel.getScores()
obsQuality = pcaModel.getObsQuality()
beta = pcaModel.getBeta()
cumulative = pcaModel.getCommun()

factorLoadings_df = pd.DataFrame(factorLoadings, index=variables,
                                 columns=('C' + str(k + 1) for k in range(numberObsVariables)))
factorLoadings_df.to_csv(path_or_buf='./dataOUT/FactorLoadings.csv')

scores_df = pd.DataFrame(scores, index=obs,
                         columns=('Comp' + str(k + 1) for k in range(numberObsVariables)))
scores_df.to_csv(path_or_buf='./dataOUT/Scores.csv')

obsQuality_df = pd.DataFrame(obsQuality, index=obs,
                             columns=('Comp' + str(k + 1) for k in range(numberObsVariables)))
obsQuality_df.to_csv(path_or_buf='./dataOUT/ObsQuality.csv')

beta_df = pd.DataFrame(beta, index=obs,
                       columns=('Comp' + str(k + 1) for k in range(numberObsVariables)))
beta_df.to_csv(path_or_buf='./dataOut/Beta.csv')

commun_df = pd.DataFrame(cumulative, index=variables,
                         columns=('C' + str(k + 1) for k in range(numberObsVariables)))
commun_df.to_csv(path_or_buf='./dataOUT/Communalities.csv')

R_df = pd.DataFrame(correlation, index=variables, columns=variables)
R_df.to_csv(path_or_buf='./dataOUT/R.csv')

# correlogram for factor loadings
graphics.correlogram(factorLoadings_df)
plt.savefig('./graphsPCA/correlogram')

# correlogram of correlation matrix
graphics.correlogram(R_df, valmin=-1, valmax=1,
                     title='Correlation matrix of causal variables')
plt.savefig('./graphsPCA/correlation_matrix_causal_variables')

# the principal components graphic
graphics.plot_variance(eigenvalues)
plt.savefig('./graphsPCA/principal_components')

# quality of the observations representation
graphics.intensity_map(obsQuality_df.iloc[:, :], dec=2,
                       title='Quality of observations on the axes of the principal components')
plt.savefig('./graphsPCA/quality_observations')

# graphic of intensity of factorial scores
graphics.intensity_map(scores_df.iloc[:, :], dec=2, title="Intensity of factorial scores")
plt.savefig('./graphsPCA/intensity_scores')

# graph of communalities
graphics.intensity_map(commun_df, dec=1,
                       title='Communalities of the principal components found in the causal variables')
plt.savefig('./graphsPCA/communalities')

graphics.show()
