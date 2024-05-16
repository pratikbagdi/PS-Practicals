# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Function to plot the clusters
def plot_clusters(X, labels, centers):
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('K-means Clustering')
    plt.show()

# Function to perform K-means clustering
def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    return labels, centers

# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
for n_clusters in range(2, 11):
    labels, _ = kmeans_clustering(X, n_clusters)
    silhouette_scores.append(silhouette_score(X, labels))

optimal_n_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 because range started from 2

# Perform K-means clustering with the optimal number of clusters
labels, centers = kmeans_clustering(X, optimal_n_clusters)

# Plot the clusters
plot_clusters(X[:, :2], labels, centers)
