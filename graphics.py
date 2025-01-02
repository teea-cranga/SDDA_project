import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


# region PCA
def correlogram(R2, dec=2, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(R2, dec), vmin=valmin, vmax=valmax,
               cmap='bwr', annot=True)


def plot_variance(alpha, title='Eigenvalues - '
                               'explained variance by the principal components',
                  labelX='Principal components',
                  labelY='Eigenvalues (explained variance)'):
    plt.figure(title,
               figsize=(11, 8))
    plt.title(title,
              fontsize=16, color='k', verticalalignment='bottom')
    plt.xlabel(labelX,
               fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(labelY,
               fontsize=14, color='k', verticalalignment='bottom')
    x_index = ['C' + str(k + 1) for k in range(len(alpha))]
    plt.plot(x_index, alpha, 'bo-')
    plt.xticks(x_index)
    plt.axhline(1, color='r')


def intensity_map(R2, dec=1, title='Intensity Map', valmin=None, valmax=None, ):
    plt.figure(title, figsize=(18, 14))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(R2, dec), vmin=valmin, vmax=valmax,
               cmap='Oranges', annot=True)


def show():
    plt.show()
# endregion

# region [insert second analysis here]

# endregion
