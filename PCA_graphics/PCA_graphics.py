import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd


def correlogram(R2, dec=2, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(R2, dec), vmin=valmin, vmax=valmax,
                cmap='bwr', annot=True)

def show():
    plt.show()