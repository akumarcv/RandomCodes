import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

EPSILON = 1e-6


class LogisticRegression:
    def __init__(self, D=10, max_iter=5000, alpha=0.01, Lambda=0.01):
        self.W = np.random.randn(D + 1, 1)
        self.bias = np.zeros(1)
        self.alpha = alpha
        self.Lambda = Lambda
        self.max_iter = max_iter

    def prediction(self, x):
        return np.matmul(x, self.W)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def error(self, predictions, y):
        N = predictions.shape[0]
        E = (-1 / N) * (
            np.sum(
                y * np.log(predictions + EPSILON)
                + (1 - y) * np.log(1 - predictions + EPSILON)
            )
        )
        regularization = self.Lambda * np.linalg.norm(self.W)
        return E + regularization

    def gradient(self, x, predictions, y):
        return (1 / x.shape[0]) * np.matmul(
            x.T, (predictions - y)
        ) + 2 * self.Lambda * self.W

    def fit(self, x_train, y_train, x_val, y_val):
        train_losses = []
        val_losses = []
        for i in range(self.max_iter):
            predictions_train = self.sigmoid(self.prediction(x_train))
            loss_train = self.error(predictions_train, y_train)
            self.W = self.W - self.alpha * self.gradient(
                x_train, predictions_train, y_train
            )
            train_losses.append(loss_train)

            predictions_val = self.sigmoid(self.prediction(x_val))
            loss_val = self.error(predictions_val, y_val)
            val_losses.append(loss_val)

            if i % 100 == 0:
                print(f"Iter {i} Train Error {loss_train} Val Error {loss_val}")
        return train_losses, val_losses


# Driver code
if __name__ == "__main__":
    # Generate synthetic dataset
    np.random.seed(42)
    data_dim = 50
    X, y = make_classification(
        n_samples=1000,
        n_features=data_dim,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    y = y.reshape(-1, 1)

    # Add bias term to X
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train the logistic regression model
    model = LogisticRegression(D=data_dim, max_iter=2000, alpha=0.01, Lambda=0.01)
    train_losses, val_losses = model.fit(X_train, y_train, X_val, y_val)

    # Plot training and validation loss vs iterations
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=range(len(train_losses)), y=train_losses, label="Training Loss", color="b"
    )
    sns.lineplot(
        x=range(len(val_losses)), y=val_losses, label="Validation Loss", color="r"
    )
    plt.fill_between(range(len(train_losses)), train_losses, alpha=0.2, color="b")
    plt.fill_between(range(len(val_losses)), val_losses, alpha=0.2, color="r")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Iterations")
    plt.legend()
    plt.savefig('logistic_regression_plots.png')  # Save the plot as a PNG file
    plt.show()
    plt.show()
