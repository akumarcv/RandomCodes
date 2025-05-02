import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

epsilon = 1e-6

def most_common(arr):
    counts = Counter(arr)
    max_count = 0
    max_element = None
    for element, count in counts.items():
        if count > max_count:
            max_count = count 
            max_element = element 
    return max_element 

def weighted_labels(arr, weights):
    weighted_labels = {}
    for element, weight in zip(arr, weights):
        if element in weighted_labels:
            weighted_labels[element] += weight
        else:
            weighted_labels[element] = weight
    
    max_vote = 0
    max_element = None
    for element, vote in weighted_labels.items():
        if vote > max_vote:
            max_vote = vote
            max_element = element
    return max_element

class KNN:
    def __init__(self, X, y, k=3):
        self.X = X
        self.y = y
        self.k = k
    
    def _calculate_weights(self, distances):
        weights = 1/(distances + epsilon)
        return weights

    def predict(self, x):
        distances = np.linalg.norm(self.X - x, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = self.y[k_indices]
        k_distances = distances[k_indices]
        weights = self._calculate_weights(k_distances)

        # return weighted_labels
        return weighted_labels(k_nearest, weights)

    def predict_batch(self, x):
        x_test = x[:, np.newaxis]
        diff = (x_test - self.X)
        distances = np.linalg.norm(diff, axis=2)
        
        indices = np.argsort(distances, axis=1)[:, :self.k]
        result = []
        for i in range(indices.shape[0]):
            label_test = most_common(indices[i, :])
            result.append(self.y[label_test])
        return np.array(result)

def visualize_results(X_train, y_train, X_test, test_labels, dataset_name):
    plt.figure(figsize=(12, 6))
    
    # Plot only the first two dimensions if more exist
    if X_train.shape[1] > 2:
        X_train_2d = X_train[:, :2]
        X_test_2d = X_test[:, :2]
    else:
        X_train_2d = X_train
        X_test_2d = X_test
    
    # Plot training data
    plt.subplot(1, 2, 1)
    unique_labels = np.unique(y_train)
    for label in unique_labels:
        plt.scatter(
            X_train_2d[y_train == label, 0],
            X_train_2d[y_train == label, 1],
            label=f'Class {label} (train)',
            alpha=0.6
        )
    plt.title(f'Training Data - {dataset_name}')
    plt.legend()
    
    # Plot test data with predictions
    plt.subplot(1, 2, 2)
    for label in unique_labels:
        plt.scatter(
            X_test_2d[test_labels == label, 0],
            X_test_2d[test_labels == label, 1],
            label=f'Class {label} (predicted)',
            alpha=0.8,
            marker='x'
        )
    plt.title('Test Data with KNN Predictions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/amitkumar/Desktop/Preparation/ml_codes/weighted_knn_results.png')
    plt.show()

def main():
    # Load dataset from sklearn
    print("Loading dataset...")
    dataset_choice = 'iris'  # You can change to 'wine', 'breast_cancer', etc.
    
    if dataset_choice == 'iris':
        dataset = datasets.load_iris()
    elif dataset_choice == 'wine':
        dataset = datasets.load_wine()
    elif dataset_choice == 'breast_cancer':
        dataset = datasets.load_breast_cancer()
    else:
        dataset = datasets.load_iris()
    
    X = dataset.data
    y = dataset.target
    
    # Split data into labeled and unlabeled (for simulation)
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create KNN model
    print("Training weighted KNN model...")
    k_value = 5
    knn = KNN(X_labeled, y_labeled, k=k_value)
    
    # Make predictions
    print("Making predictions on test data...")
    individual_predictions = []
    for i in range(len(X_unlabeled)):
        label = knn.predict(X_unlabeled[i])
        individual_predictions.append(label)
        
    batch_predictions = knn.predict_batch(X_unlabeled)
    
    print(f"Individual predictions for first 5 samples: {individual_predictions[:5]}")
    print(f"Batch predictions for first 5 samples: {batch_predictions[:5]}")
    
    # Visualize results
    print("Generating visualization...")
    visualize_results(X_labeled, y_labeled, X_unlabeled, 
                    np.array(individual_predictions), dataset.DESCR.splitlines()[0])
    
    print("Done! Visualization saved to ml_codes/weighted_knn_results.png")

if __name__ == "__main__":
    main()