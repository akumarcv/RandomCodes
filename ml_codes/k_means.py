import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler


class KMeans:
    """
    K-Means clustering algorithm implementation.

    This class implements the standard K-means clustering algorithm that partitions
    data points into K clusters, where each data point belongs to the cluster with
    the nearest centroid.

    Parameters:
    -----------
    X : np.ndarray
        Input data as a numpy array of shape (n_samples, n_features)
    k : int
        Number of clusters to form
    max_iterations : int
        Maximum number of iterations for the algorithm to converge

    Attributes:
    -----------
    centroids : np.ndarray
        Coordinates of cluster centers
    assigned_clusters : np.ndarray
        Cluster index for each data point
    """

    def __init__(self, X: np.ndarray, k: int, max_iterations: int):
        self.X = X
        self.centroids = []
        self.K = k
        self.max_interations = max_iterations
        self.assigned_clusters = np.zeros(self.X.shape[0])

    def cluster(self):
        """
        Performs K-means clustering on the input data.

        The algorithm follows these steps:
        1. Initialize centroids randomly
        2. Assign each data point to the nearest centroid
        3. Update centroids based on assigned points
        4. Repeat steps 2-3 until convergence or max iterations reached

        Returns:
        --------
        None : Updates the centroids and assigned_clusters attributes in-place
        """
        # Initialize centroids by randomly selecting K data points from the dataset
        random_indices = np.random.randint(0, self.X.shape[0], self.K)
        self.centroids = self.X[random_indices]

        # Main K-means loop
        for _ in range(self.max_interations):
            # Store previous centroids to check for convergence
            previous_clusters = self.centroids.copy()

            # Step 1: Assign each data point to the nearest centroid
            for i, point in enumerate(self.X):
                # Calculate Euclidean distance from point to each centroid
                distances_with_centroids = [
                    np.linalg.norm(point - self.centroids[j, :]) for j in range(self.K)
                ]
                # Get the index of the closest centroid
                lowest_distance = np.argmin(distances_with_centroids)
                self.assigned_clusters[i] = lowest_distance

            # Step 2: Update centroids based on the mean of assigned data points
            movement = 0  # Track how much centroids move to check for convergence
            for k in range(self.K):
                # Check if the cluster has any points
                if np.sum(self.assigned_clusters == k) > 0:
                    # Update centroid as the mean of all points assigned to this cluster
                    self.centroids[k] = np.mean(
                        self.X[self.assigned_clusters == k], axis=0
                    )
                # If no points in cluster, leave centroid as is (or optionally reinitialize it)

                # Calculate total movement of this centroid
                movement = movement + np.linalg.norm(
                    previous_clusters[k] - self.centroids[k]
                )

            # Convergence check: if centroids barely moved, we're done
            if movement < 1e-6:
                break


def plot_dataset(X, y=None, title="Dataset"):
    """Plot the original dataset with true labels if available."""
    plt.figure(figsize=(10, 7))
    if y is not None:
        plt.scatter(
            X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, alpha=0.8, edgecolors="w"
        )
    else:
        plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.8, edgecolors="w")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_clustering_results(X, labels, centroids, title="Clustering Results"):
    """Plot the clustering results with different colors for each cluster."""
    plt.figure(figsize=(10, 7))

    # Plot data points colored by cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=50,
            c=[colors[i]],
            label=f"Cluster {int(label)}",
            alpha=0.7,
            edgecolors="w",
        )

    # Plot centroids
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        marker="X",
        c="red",
        edgecolors="black",
        linewidth=2,
        label="Centroids",
    )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create datasets for clustering
    datasets = [
        {
            "name": "Blobs",
            "data": make_blobs(
                n_samples=300, centers=4, cluster_std=0.6, random_state=42
            ),
            "k": 4,
        },
        {
            "name": "Anisotropic Blobs",
            "data": make_blobs(n_samples=300, centers=3, random_state=42),
            "k": 3,
        },
        {
            "name": "Moons",
            "data": make_moons(n_samples=300, noise=0.1, random_state=42),
            "k": 2,
        },
    ]

    for dataset in datasets:
        name = dataset["name"]
        X, y_true = dataset["data"]
        k = dataset["k"]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Plot original dataset with true labels
        plot_dataset(X_scaled, y_true, f"Original {name} Dataset")

        # Apply our K-means implementation
        kmeans = KMeans(X=X_scaled, k=k, max_iterations=100)
        kmeans.cluster()

        # Plot clustering results
        plot_clustering_results(
            X_scaled,
            kmeans.assigned_clusters,
            kmeans.centroids,
            f"K-means Clustering on {name} (K={k})",
        )

        # Print some stats
        print(f"Dataset: {name}")
        print(f"Number of samples: {X.shape[0]}")
        print(f"Number of clusters: {k}")
        print(f"Cluster sizes:")
        for i in range(k):
            print(f"  Cluster {i}: {np.sum(kmeans.assigned_clusters == i)} points")
        print("-" * 50)

    plt.show()


if __name__ == "__main__":
    main()
