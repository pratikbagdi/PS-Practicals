# Import libraries
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()

# Data (features) and target (species)
data = iris.data
target = iris.target

# Define the number of clusters (k)
k = 3

# Create the K-Means model
kmeans = KMeans(n_clusters=k, n_init=10)  # Set n_init to 10 explicitly

# Train the model (fit the data)
kmeans.fit(data)

# Get cluster labels for each data point
cluster_labels = kmeans.labels_

# Plot the data with colors corresponding to clusters
plt.scatter(data[:, 0], data[:, 1], c=cluster_labels)  # Using first two features for visualization

# Add labels for each cluster (optional)
for i, label in enumerate(kmeans.cluster_centers_):
    plt.scatter(label[0], label[1], color='black', marker='s', label=f'Cluster {i+1}')

# Add legend and title
plt.title('Iris Dataset - K-Means Clustering')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.legend()

# Show the plot
plt.show()

# Print the target values for reference (optional)
print("Actual species:")
print(target)

# Print the cluster labels for each data point
print("Predicted cluster labels:")
print(cluster_labels)