import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegression:
    def __init__(self, alpha, max_iter=1000, Lambda=0.001):
        self.alpha = alpha
        self.max_iter = max_iter
        self.W = None

        self.val_losses = []
        self.train_losses = []
        self.val_accuracy = []
        self.train_accuracy = []
        self.Lambda = Lambda

    def initialize_weights(self, n_features):
        """Initialize weights randomly"""
        self.W = np.random.randn(n_features) * 0.01

    def error(self, predictions, y):
        N = predictions.shape[0]
        E = (1 / N) * np.sum((predictions - y) ** 2) 
        regularization = self.Lambda * np.linalg.norm(self.W)
        return E + regularization

    def gradient(self, x, y, predictions):
        N = x.shape[0]
        return (
            2 / N * np.matmul(x.T, (predictions - y))
        ) + 2 * self.Lambda * self.W

    def fit(self, x_train, y_train, x_val, y_val):
        self.initialize_weights(x_train.shape[1])

        for i in range(self.max_iter):
            predictions_train = self.prediction(x_train)
            loss_train = self.error(predictions_train, y_train)
            self.W = self.W - self.alpha * self.gradient(x_train, y_train, predictions_train)
            self.train_losses.append(loss_train)
            predictions_val = self.prediction(x_val)
            loss_val = self.error(predictions_val, y_val)
            self.val_losses.append(loss_val)
            if i % 100 == 0:
                print(f"Iter {i} Train Error {loss_train} Val Error {loss_val}")
        return self.train_losses, self.val_losses

    def prediction(self, x):
        return np.matmul(x, self.W)

    def predict(self, x):
        """Make predictions using the current weights"""
        return np.dot(x, self.W)


# Import our linear regression implementation
from linear_regression import LinearRegression


def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load regression dataset
    # Option 1: California Housing dataset (multiple features)
    # dataset = fetch_california_housing()

    # Option 2: Diabetes dataset (multiple features)
    # dataset = load_diabetes()

    # Option 3: Create a simple synthetic dataset (better for visualization)
    X, y, coef = make_regression(
        n_samples=200,
        n_features=1,  # Using just one feature for easy visualization
        noise=20,
        coef=True,  # Return the coefficients of the underlying linear model
        random_state=42,
    )

    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42
    )  # 0.25 x 0.8 = 0.2 of the original data

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create and train linear regression model
    model = LinearRegression(alpha=0.01, max_iter=1000, Lambda=0.001)
    train_losses, val_losses = model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

    # Make predictions on test set
    y_pred = model.predict(X_test_scaled)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test Mean Squared Error: {mse:.4f}")
    print(f"Test R² Score: {r2:.4f}")
    print(f"True coefficients: {coef}")
    print(f"Learned coefficients: {model.W}")

    # Plot training and validation losses
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot regression line vs. actual data
    plt.subplot(1, 2, 2)

    # Sort for better visualization of the regression line
    sort_idx = np.argsort(X_test.flatten())

    # Plot original data points
    plt.scatter(X_test, y_test, color="blue", alpha=0.6, label="Actual Data")

    # Plot regression line
    plt.plot(
        X_test[sort_idx],
        y_pred[sort_idx],
        color="red",
        linewidth=2,
        label="Regression Line",
    )

    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title(f"Linear Regression (R² = {r2:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plt.savefig("linear_regression_results.png", dpi=300, bbox_inches="tight")
    print("Plot saved as 'linear_regression_results.png'")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
