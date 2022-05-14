# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('test_folder/SOL-USD_mod.csv')
x = dataset.iloc[:, 5:-3].values

# Using the dendrogram to find the optimal number of clusters     # Result : 3 clusters
# import scipy.cluster.hierarchy as sch
# dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))      # ward = minimum variant
# plt.title("Dendrogram")
# plt.xlabel("customers")
# plt.ylabel("euclidian distance")
# plt.show()

# Take 3 clusters
# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(x)
print(y_hc)

new_data = dataset
new_data["clustering result"] = y_hc
new_data.to_csv(path_or_buf='test_folder/SOL-USD_mod_hie.csv')
#
# # Visualising the clusters
# plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s=30, c="red", label="Cluster 0")
# plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s=30, c="blue", label="Cluster 1")
# plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s=30, c="green", label="Cluster 2")
# # plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s=30, c="cyan", label="Cluster 3")
# # plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s=30, c="magenta", label="Cluster 4")
# # plt.scatter(y_hc.cluster_centers_[:, 0], y_hc.cluster_centers_[:, 1], s=100, c="yellow", label="Centroids")
# plt.title("Clusters of customers")
# plt.xlabel("Annual income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.legend()
# plt.show()