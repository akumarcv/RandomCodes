import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import argparse
import sys

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
    def __init__(self, X, y, k=3, weight_function='inverse', sigma=1.0):
        self.X = X
        self.y = y
        self.k = k
        self.weight_function = weight_function
        self.sigma = sigma
    
    def _calculate_weights(self, distances):
        """
        Calculate weights for neighbors based on their distances.
        
        Parameters:
        distances : np.ndarray
            Distances to each neighbor
            
        Returns:
        np.ndarray : Weights for each neighbor
        """
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        
        if self.weight_function == 'inverse':
            # Weight is inversely proportional to distance
            weights = 1.0 / (distances + epsilon)
        elif self.weight_function == 'inverse_squared':
            # Weight is inversely proportional to the square of distance
            weights = 1.0 / ((distances ** 2) + epsilon)
        elif self.weight_function == 'gaussian':
            # Weight follows a Gaussian curve based on distance
            weights = np.exp(-(distances ** 2) / (2 * (self.sigma ** 2)))
        else:
            # Default to equal weights if function not recognized
            weights = np.ones_like(distances)
            
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

def visualize_results(X_train, y_train, X_test, test_labels, dataset_name, output_file=None):
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
    if output_file:
        plt.savefig(output_file)
    plt.show()

def compare_weighting_schemes(X_train, y_train, X_test, y_test, dataset_name, k=5):
    """
    Compare different weighting schemes and visualize their results
    
    Parameters:
    X_train, y_train: Training data and labels
    X_test, y_test: Testing data and labels
    dataset_name: Name of the dataset
    k: Number of neighbors to use
    """
    weight_functions = ['uniform', 'inverse', 'inverse_squared', 'gaussian']
    
    # Create a figure with subplots for each weighting scheme
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Comparison of Weighting Schemes for {dataset_name} Dataset (k={k})', fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    results = {}
    
    # For each weighting scheme
    for i, weight_func in enumerate(weight_functions):
        # Train KNN model with the current weighting scheme
        knn = KNN(X_train, y_train, k=k, weight_function=weight_func, sigma=1.0)
        
        # Make predictions
        predictions = []
        for sample in X_test:
            pred = knn.predict(sample)
            predictions.append(pred)
        predictions = np.array(predictions)
        
        # Calculate accuracy
        accuracy = np.mean(predictions == y_test) * 100
        results[weight_func] = accuracy
        
        # Plot results
        ax = axes[i]
        
        # Plot only the first two dimensions
        if X_train.shape[1] > 2:
            X_train_2d = X_train[:, :2]
            X_test_2d = X_test[:, :2]
        else:
            X_train_2d = X_train
            X_test_2d = X_test
        
        # Plot training data (smaller points)
        unique_labels = np.unique(y_train)
        for label in unique_labels:
            ax.scatter(
                X_train_2d[y_train == label, 0],
                X_train_2d[y_train == label, 1],
                label=f'Class {label} (train)',
                alpha=0.3,
                s=30
            )
        
        # Plot test data predictions (larger markers)
        for label in unique_labels:
            ax.scatter(
                X_test_2d[predictions == label, 0],
                X_test_2d[predictions == label, 1],
                label=f'Class {label} (pred)',
                alpha=0.7,
                marker='x',
                s=50
            )
        
        # Highlight misclassified points with a red circle
        misclassified = predictions != y_test
        ax.scatter(
            X_test_2d[misclassified, 0],
            X_test_2d[misclassified, 1],
            facecolors='none',
            edgecolors='red',
            s=100,
            label='Misclassified'
        )
        
        ax.set_title(f'{weight_func.capitalize()} Weighting (Accuracy: {accuracy:.2f}%)')
        ax.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the super title
    
    # Save the figure
    output_file = '/Users/amitkumar/Desktop/Preparation/ml_codes/knn_weighting_comparison.png'
    plt.savefig(output_file)
    plt.show()
    
    # Print accuracy comparison
    print("\nAccuracy Comparison for Different Weighting Schemes:")
    print("=" * 50)
    for weight_func, accuracy in results.items():
        print(f"{weight_func.capitalize():15s}: {accuracy:.2f}%")
    print("=" * 50)
    
    # Return the best weighting scheme
    best_scheme = max(results, key=results.get)
    print(f"\nBest weighting scheme: {best_scheme} with accuracy: {results[best_scheme]:.2f}%")
    
    return best_scheme, results

def parse_arguments():
    """
    Parse command line arguments for the KNN script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Weighted KNN Classification')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='iris',
                        choices=['iris', 'wine', 'breast_cancer', 'digits'],
                        help='Dataset to use for classification')
    
    # Model arguments
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors to use')
    parser.add_argument('--weight-function', type=str, default='inverse',
                        choices=['uniform', 'inverse', 'inverse_squared', 'gaussian'],
                        help='Weighting function for neighbors')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Sigma parameter for Gaussian weighting function')
    
    # Visualization arguments
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting of results')
    parser.add_argument('--output-file', type=str, 
                        default='/Users/amitkumar/Desktop/Preparation/ml_codes/weighted_knn_results.png',
                        help='Path to save the visualization')
    
    # Add comparison flag
    parser.add_argument('--compare-weights', action='store_true',
                        help='Compare different weighting schemes on the same dataset')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Load dataset from sklearn
    print(f"Loading dataset: {args.dataset}...")
    
    if args.dataset == 'iris':
        dataset = datasets.load_iris()
    elif args.dataset == 'wine':
        dataset = datasets.load_wine()
    elif args.dataset == 'breast_cancer':
        dataset = datasets.load_breast_cancer()
    elif args.dataset == 'digits':
        dataset = datasets.load_digits()
        # For digits, we'll use a subset to improve performance
        X = dataset.data[:500]
        y = dataset.target[:500]
        dataset_name = 'Digits Dataset'
    else:
        print(f"Unknown dataset: {args.dataset}, falling back to iris")
        dataset = datasets.load_iris()
    
    # Set X and y for non-digits datasets
    if args.dataset != 'digits':
        X = dataset.data
        y = dataset.target
        dataset_name = dataset.DESCR.splitlines()[0]
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Split data into labeled and unlabeled (for simulation)
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # If comparing different weighting schemes
    if args.compare_weights:
        print("\nComparing different weighting schemes...")
        best_scheme, results = compare_weighting_schemes(
            X_labeled, y_labeled, X_unlabeled, y_unlabeled, dataset_name, k=args.k
        )
        print(f"Visualization saved to: /Users/amitkumar/Desktop/Preparation/ml_codes/knn_weighting_comparison.png")
        # Set the weight function to the best one for the individual run
        args.weight_function = best_scheme
        print(f"\nUsing best weighting scheme ({best_scheme}) for individual run...")
    
    # Create KNN model
    print(f"\nTraining weighted KNN model with k={args.k} and weight function={args.weight_function}...")
    knn = KNN(
        X_labeled, 
        y_labeled, 
        k=args.k, 
        weight_function=args.weight_function,
        sigma=args.sigma
    )
    
    # Make predictions
    print("Making predictions on test data...")
    individual_predictions = []
    for i in range(len(X_unlabeled)):
        label = knn.predict(X_unlabeled[i])
        individual_predictions.append(label)
    
    individual_predictions = np.array(individual_predictions)
    
    # Calculate accuracy
    accuracy = np.mean(individual_predictions == y_unlabeled) * 100
    print(f"Prediction accuracy: {accuracy:.2f}%")
    
    # Visualize results if not disabled
    if not args.no_plot and not args.compare_weights:
        print("Generating visualization...")
        visualize_results(
            X_labeled, 
            y_labeled, 
            X_unlabeled, 
            individual_predictions, 
            dataset_name,
            args.output_file
        )
        print(f"Visualization saved to {args.output_file}")
    
    print("Done!")

if __name__ == "__main__":
    main()