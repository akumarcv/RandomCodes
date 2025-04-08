import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



class KNN:
    def __init__(self, X, y, k=3):
        """
        Initialize the KNN classifier with the training data, labels, and number of neighbors.

        Parameters:
        X : np.ndarray
            Training data of shape (n_samples, n_features)
        y : np.ndarray
            Training labels of shape (n_samples,)
        k : int
            Number of nearest neighbors to consider for classification
        """
        self.X = X
        self.y = y
        self.k = k

    def predict(self, x):
        """
        Predict the class of a new data point based on the k nearest neighbors.

        Parameters:
        x : np.ndarray
            New data point of shape (n_features,)

        Returns:
        int : Predicted class label
        """
        # Calculate distances from the new point to all training points
        distances = np.linalg.norm(self.X - x, axis=1)
        
        # Get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = self.y[k_indices]
        
        # Return the most common label among the k nearest neighbors
        return np.bincount(k_nearest_labels).argmax()
    
    def predict_batch(self, X_test):
        """
        Predict classes for multiple test samples.
        
        Parameters:
        X_test : np.ndarray
            Test data of shape (n_samples, n_features)
            
        Returns:
        np.ndarray : Predicted class labels for each test sample
        """
        return np.array([self.predict(x) for x in X_test])
    

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plot the decision boundary of a classifier.
    
    Parameters:
    model : Model with predict method
    X : np.ndarray - Training data
    y : np.ndarray - Training labels
    title : str - Plot title
    """
    h = 0.02  # Step size in the mesh
    
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh grid points
    Z = model.predict_batch(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and training points
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.rainbow)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.tight_layout()

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create datasets for testing
    datasets = [
        {
            'name': 'Blobs (Linearly Separable)',
            'data': make_blobs(n_samples=300, centers=3, random_state=42),
            'k': 5
        },
        {
            'name': 'Moons (Non-linearly Separable)',
            'data': make_moons(n_samples=300, noise=0.1, random_state=42),
            'k': 7
        },
        {
            'name': 'Circles (Concentric)',
            'data': make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42),
            'k': 9
        }
    ]
    
    for dataset in datasets:
        name = dataset['name']
        X, y = dataset['data']
        k = dataset['k']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Create and train KNN model
        knn = KNN(X_train, y_train, k=k)
        
        # Make predictions on test set
        y_pred = knn.predict_batch(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Dataset: {name}")
        print(f"k = {k}")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("-" * 50)
        
        # Plot original data
        plt.figure(figsize=(10, 7))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.rainbow, edgecolors='k')
        plt.title(f"Original {name} Dataset")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar()
        plt.tight_layout()
        
        # Plot decision boundary
        plot_decision_boundary(knn, X_train, y_train, 
                              title=f"KNN Decision Boundary on {name} (k={k})")
    
    plt.show()

if __name__ == "__main__":
    main()