import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def median_filter(X, kernel_size, stride=2):
    pad = kernel_size // 2
    X_padded = np.pad(X, pad, mode="constant", constant_values=0)

    output_height = (X_padded.shape[0] - kernel_size) // stride + 1
    output_width = (X_padded.shape[1] - kernel_size) // stride + 1
    result_X = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            values = X_padded[
                i * stride : i * stride + kernel_size,
                j * stride : j * stride + kernel_size,
            ]
            median = np.median(values)
            result_X[i, j] = median
    return result_X

def mean_filter(X, kernel_size, stride=2):
    pad = kernel_size // 2
    X_padded = np.pad(X, pad, mode="constant", constant_values=0)

    output_height = (X_padded.shape[0] - kernel_size) // stride + 1
    output_width = (X_padded.shape[1] - kernel_size) // stride + 1
    result_X = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            values = X_padded[
                i * stride : i * stride + kernel_size,
                j * stride : j * stride + kernel_size,
            ]
            mean = np.mean(values)
            result_X[i, j] = mean
    return result_X

# Driver code
if __name__ == "__main__":
    image = True
    random_data = True

    if image: 
        # read image
        X = cv2.imread("512-grayscale-image-Cameraman.png", cv2.IMREAD_GRAYSCALE).astype(int)

        # Apply median filter
        kernel_size = 7
        stride = 3
        result_median = median_filter(X, kernel_size, stride)

        # Apply mean filter
        result_mean = mean_filter(X, kernel_size, stride)

        # Plot the input and output arrays in heatmap form
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

        # Plot input array
        sns.heatmap(
            X.astype(int),
            annot=False,
            fmt="d",
            cmap="viridis",
            ax=ax1,
            cbar=False,
            annot_kws={"size": 6},
        )
        ax1.set_title("Original Image")

        # Plot median filter output array
        sns.heatmap(
            result_median.astype(int),
            annot=False,
            fmt="d",
            cmap="viridis",
            ax=ax2,
            cbar=False,
            annot_kws={"size": 6},
        )
        ax2.set_title("Median Filtered Image")

        # Plot mean filter output array
        sns.heatmap(
            result_mean.astype(int),
            annot=False,
            fmt="d",
            cmap="viridis",
            ax=ax3,
            cbar=False,
            annot_kws={"size": 6},
        )
        ax3.set_title("Mean Filtered Image")

        plt.tight_layout()
        plt.savefig("mean_median_filter_image.png")
        plt.show()

    if random_data:
        # Create a random 50x50 2D array with positive values <255 and >0
        np.random.seed(42)
        X = np.random.randint(1, 256, size=(50, 50))

        # Apply median filter
        kernel_size = 5
        stride = 2
        result_median = median_filter(X, kernel_size, stride)

        # Apply mean filter
        result_mean = mean_filter(X, kernel_size, stride)

        # Plot the input and output arrays in heatmap form
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

        # Plot input array
        sns.heatmap(
            X.astype(int),
            annot=True,
            fmt="d",
            cmap="viridis",
            ax=ax1,
            cbar=False,
            annot_kws={"size": 6},
        )
        ax1.set_title("Input Array")

        # Plot median filter output array
        sns.heatmap(
            result_median.astype(int),
            annot=True,
            fmt="d",
            cmap="viridis",
            ax=ax2,
            cbar=False,
            annot_kws={"size": 6},
        )
        ax2.set_title("Median Filtered Array")

        # Plot mean filter output array
        sns.heatmap(
            result_mean.astype(int),
            annot=True,
            fmt="d",
            cmap="viridis",
            ax=ax3,
            cbar=False,
            annot_kws={"size": 6},
        )
        ax3.set_title("Mean Filtered Array")

        plt.tight_layout()
        plt.savefig("mean_median_filter_random.png")
        plt.show()