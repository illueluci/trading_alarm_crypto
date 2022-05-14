# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('test_folder/SOL-USD_mod.csv')
x = dataset.iloc[:, 5:-3].values

print(x)

# Using the elbow method to find the optimal number of clusters (looks like 6...)
from sklearn.cluster import KMeans
# wcss = []                                                 # python list, empty
# for i in range(1, 20+1):
#     kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
#     kmeans.fit(x)
#     wcss.append(kmeans.inertia_)
# # plot the wcss to actually use the elbow method
# plt.plot(range(1, 20+1), wcss)
# plt.title("The Elbow Method")
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.show()

# Training the K-Means model on the dataset
actual_kmeans = KMeans(n_clusters=6, init="k-means++", random_state=42)
y_kmeans = actual_kmeans.fit_predict(x)
print(y_kmeans)

new_data = dataset
new_data["clustering result"] = y_kmeans
new_data.to_csv(path_or_buf='test_folder/SOL-USD_mod_kmeans.csv')

# Visualising the clusters
# plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=30, c="red", label="Cluster 0")
# plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=30, c="blue", label="Cluster 1")
# plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=30, c="green", label="Cluster 2")
# plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=30, c="cyan", label="Cluster 3")
# plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=30, c="magenta", label="Cluster 4")
# plt.scatter(actual_kmeans.cluster_centers_[:, 0], actual_kmeans.cluster_centers_[:, 1], s=100, c="yellow", label="Centroids")
# plt.title("Clusters of customers")
# plt.xlabel("Annual income (k$)")
# plt.ylabel("Spending Score (1-100)")
# plt.legend()
# plt.show()