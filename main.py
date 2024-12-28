import pandas as pd
import PCA.PCA as pca
import PCA_graphics.PCA_graphics as graphics
import matplotlib.pyplot as plt

# extract the data from the .csv file in a pandas.DataFrame
table = pd.read_csv("./dataIN/depression_data.csv")
print(table)

obs = table.index[:]
vars = table.columns[:]
X = table[vars].values

numberObsVariables = X.shape[1]

pcaModel = pca.PCA(X)
correlation = pcaModel.getCorr()
eigenvalues = pcaModel.getEigenvalues()
eigenvectors = pcaModel.getEigenvectors()
factorLoadings = pcaModel.getFactorLoadings()
components = pcaModel.getComponents()
scores = pcaModel.getScores()
obsQuality = pcaModel.getObsQuality()
beta = pcaModel.getBeta()
cumulative = pcaModel.getCommun()

factorLoadings_df = pd.DataFrame(factorLoadings, index=vars,
                       columns=('C'+str(k+1) for k in range(numberObsVariables)))
factorLoadings_df.to_csv(path_or_buf='./dataOUT/FactorLoadings.csv')

# correlogram for factor loadings
graphics.correlogram(factorLoadings_df)
plt.savefig('./dataOUT/Correlogram')

scores_df = pd.DataFrame(scores, index=obs,
                       columns=('Comp'+str(k+1) for k in range(numberObsVariables)))
scores_df.to_csv(path_or_buf='./dataOUT/Scores.csv')

obsQuality_df = pd.DataFrame(obsQuality, index=obs,
                       columns=('Comp'+str(k+1) for k in range(numberObsVariables)))
obsQuality_df.to_csv(path_or_buf='./dataOUT/ObsQuality.csv')

beta_df = pd.DataFrame(beta, index=obs,
                       columns=('Comp'+str(k+1) for k in range(numberObsVariables)))
beta_df.to_csv(path_or_buf='./dataOut/Beta.csv')

commun_df = pd.DataFrame(cumulative, index=vars,
                       columns=('C'+str(k+1) for k in range(numberObsVariables)))
commun_df.to_csv(path_or_buf='./dataOUT/Communalities.csv')

graphics.show()




