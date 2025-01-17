import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hic
from sklearn.preprocessing import StandardScaler, LabelEncoder


def dendrogram(h, labels, title='Hierarchical Classification', threshold=None):
    """
    Plots dendrogram
    """
    plt.figure(figsize=(15, 8))
    plt.title(title, fontsize=16, color='k')
    hic.dendrogram(h, labels=labels, leaf_rotation=30)
    if threshold:
        plt.axhline(threshold, c='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
        plt.legend()
    plt.xlabel('Fraud', fontsize=14, color='k')
    plt.ylabel('Distance', fontsize=14, color='k')


def threshold(h):
    """
    Optimal threshold for cutting the dendrogram.
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


def preprocess_data(table):
    """
    Preprocess the input data: normalize numerical columns and encode categorical columns.
    """
    numerical_cols = [ 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
    categorical_cols = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order', 'fraud']

    # Normalize numerical data
    scaler = StandardScaler()
    table[numerical_cols] = scaler.fit_transform(table[numerical_cols])

    # Encode categorical data
    for col in categorical_cols:
        table[col] = LabelEncoder().fit_transform(table[col])

    return table


def hierarchical_clustering(input_file, output_dir, start_idx=0, end_idx=None, method="ward", metric="euclidean"):
    """
    Main function to perform Hierarchical Clustering Analysis (HCA).
    """
    try:
        # Load dataset
        table = pd.read_csv(input_file, na_values='')
        print(table.columns)
        # Extract the first column as labels
        labels_column = table.iloc[:, 0].values  # First column for labels
        table = table.iloc[:, 1:]  # Remove the first column from data


        # Preprocess the data
        table = preprocess_data(table)

        # Prepare data for clustering
        fraud_labels = table['fraud'].iloc[start_idx:end_idx].values  # Only for visualization
        X = table.iloc[start_idx:end_idx, :].drop(columns=['fraud']).values  # Exclude target

        # Perform Hierarchical Clustering
        HC = hic.linkage(X, method=method, metric=metric)
        t, j, m = threshold(HC)

        # Save threshold information
        with open(f'{output_dir}/result.txt', 'w') as file:
            file.write(f'Threshold = {t}\nMax Difference Junction = {j}\nNumber of Junctions = {m}')

        # Generate dendrogram
        dendrogram(HC, labels=fraud_labels, title=f'Hierarchical Classification ({method} - {metric})', threshold=t)
        plt.savefig(f'{output_dir}/hierarchical_classification.png')

        print("HCA Analysis Complete. Results saved in:", output_dir)
        plt.show()

    except Exception as e:
        print("Error during HCA analysis:", e)


if __name__ == "__main__":
    # Example usage
    input_file = "dataIN/card_transdata.csv"
    output_dir = "./dataOUT"
    start_idx = 10
    end_idx = 64
    method = "ward"                               # Linkage method (e.g., 'ward', 'single', 'complete')
    metric = "euclidean"                          # Distance metric (e.g., 'euclidean', 'manhattan')

    hierarchical_clustering(input_file, output_dir, start_idx, end_idx, method, metric)
