import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hic
import scipy.spatial.distance as dis


def dendrogram(h, labels, title='Hierarchical Classification', threshold=None):
    """
    Plots  dendrogram
    """
    plt.figure(figsize=(15, 8))
    plt.title(title, fontsize=16, color='k')
    hic.dendrogram(h, labels=labels, leaf_rotation=30)
    if threshold:
        plt.axhline(threshold, c='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.legend()
    plt.xlabel('Observations', fontsize=14, color='k')
    plt.ylabel('Distance', fontsize=14, color='k')


def threshold(h):
    """
    optimal threshold for cutting the dendrogram.
    """
    m = np.shape(h)[0]
    dist_1 = h[1:m, 2]
    dist_2 = h[0:m - 1, 2]
    diff = dist_1 - dist_2
    j = np.argmax(diff)
    threshold_value = (h[j, 2] + h[j + 1, 2]) / 2
    return threshold_value, j, m


def clusters(h, k):
    """
    Assigns observations to clusters based on the dendrogram cut.
    """
    n = np.shape(h)[0] + 1
    g = np.arange(0, n)
    for i in range(n - k):
        k1 = h[i, 0]
        k2 = h[i, 1]
        g[g == k1] = n + i
        g[g == k2] = n + i
    cat = pd.Categorical(g)
    return ['C' + str(i) for i in cat.codes], cat.codes


def hierarchical_clustering(input_file, output_dir, start_idx=0, end_idx=None, method="ward", metric="euclidean"):
    """
    Main function to perform Hierarchical Clustering Analysis (HCA).

        method (str): Linkage method for clustering.
        metric (str): Distance metric for clustering.
    """
    try:
        # Load dataset
        table = pd.read_csv(input_file, index_col=0, na_values='')
        obs_names = [int(i) for i in table.index[start_idx:end_idx]]
        X = table.iloc[start_idx:end_idx, :].values

        # Perform Hierarchical Clustering
        HC = hic.linkage(X, method=method, metric=metric)
        t, j, m = threshold(HC)

        # Save threshold information
        with open(f'{output_dir}/result.txt', 'w') as file:
            file.write(f'Threshold = {t}\nMax Difference Junction = {j}\nNumber of Junctions = {m}')

        # Generate and save dendrogram
        dendrogram(HC, labels=obs_names, title=f'Hierarchical Classification ({method} - {metric})', threshold=t)
        plt.savefig(f'{output_dir}/hierarchical_classification.png')

        # Determine clusters and save them
        k = m - j
        labels, codes = clusters(HC, k)
        ClusterTab = pd.DataFrame(data=labels, index=obs_names, columns=['Cluster'])
        ClusterTab.to_csv(f'{output_dir}/indClusters.csv')

        print("HCA Analysis Complete. Results saved in:", output_dir)
        #plt.yticks(np.arange(0, 250, 25))
        plt.yscale('symlog') #

        plt.show()

    except Exception as e:
        print("Error during HCA analysis:", e)


if __name__ == "__main__":
    # Example usage
    input_file = "dataIN/cards.csv"
    output_dir = "./dataOUT"
    start_idx = 10
    end_idx = 64
    method = "ward"                               # Linkage method (e.g., 'ward', 'single', 'complete')
    metric = "euclidean"                          # Distance metric (e.g., 'euclidean', 'manhattan')

    hierarchical_clustering(input_file, output_dir, start_idx, end_idx, method, metric)