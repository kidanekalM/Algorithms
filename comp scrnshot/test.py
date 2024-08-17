import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score

# Load the Iris dataset
data = pd.read_csv('C:/Users/PC/Desktop/New folder/slides/semester 10 - spring/cs488 AI/assing/Algorithms/Iris.csv')

# Extract the feature columns and remove any rows with missing values
X = data.iloc[:, 1:5].values
actual_labels = data.iloc[:, 5].values

# Normalize the dataset
X_norm = (X - X.mean(axis=0)) / X.std(axis=0)

# Perform k-means clustering with k=3
k = 3
centroids = np.random.rand(k, 4)
labels = np.zeros(len(X_norm))
tolerance = 1e-6
max_iterations = 1000

for _ in range(max_iterations):
    # Assign data points to clusters
    for i in range(len(X_norm)):
        distances = np.linalg.norm(X_norm[i] - centroids, axis=1)
        labels[i] = np.argmin(distances)

    # Update centroids
    new_centroids = np.zeros_like(centroids)
    for j in range(k):
        new_centroids[j] = np.mean(X_norm[labels == j], axis=0)

    # Check for convergence
    if np.linalg.norm(centroids - new_centroids) < tolerance:
        break

    centroids = new_centroids

# Visualize the cluster assignments and centroids in 2D and 3D plots
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])

# 2D plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X_norm[:, 0], X_norm[:, 1], c=labels, cmap='viridis')
ax1.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')
ax1.set_xlabel('Sepal Length (Normalized)')
ax1.set_ylabel('Sepal Width (Normalized)')
ax1.set_title('2D Iris Cluster Visualization')
ax1.legend()

# 3D plot
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
ax2.scatter(X_norm[:, 0], X_norm[:, 1], X_norm[:, 2], c=labels, cmap='viridis')
ax2.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, c='red', label='Centroids')
ax2.set_xlabel('Sepal Length (Normalized)')
ax2.set_ylabel('Sepal Width (Normalized)')
ax2.set_zlabel('Petal Length (Normalized)')
ax2.set_title('3D Iris Cluster Visualization')
ax2.legend()

plt.tight_layout()
plt.show()

# Plot the silhouette score
fig, ax = plt.subplots(figsize=(8, 6))
silhouette_avg = silhouette_score(X_norm, labels)
ax.plot(np.arange(1, 11), [silhouette_score(X_norm, np.random.randint(0, 3, len(X_norm))) for _ in range(10)], label='Random Clustering')
ax.plot([1], [silhouette_avg], marker='o', markersize=10, label='K-Means Clustering')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Silhouette Score')
ax.set_title('Silhouette Score Analysis')
ax.legend()
plt.tight_layout()
plt.show()

# Compare the cluster assignments with the actual class labels
cluster_labels = []
for label in labels:
    if label == 0:
        cluster_labels.append('Iris-setosa')
    elif label == 1:
        cluster_labels.append('Iris-versicolor')
    else:
        cluster_labels.append('Iris-virginica')

print("Comparison of Cluster Assignments and Actual Class Labels:")
for i in range(len(actual_labels)):
    print(f"Actual Label: {actual_labels[i]}, Cluster Label: {cluster_labels[i]}")

print(f"Silhouette Score: {silhouette_avg:.2f}")