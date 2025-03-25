import numpy as np
import seaborn as sns
import cv2
import matplotlib.pyplot as plt

def convolution(X, kernel, stride=2):
    kernel_h , kernel_w = kernel.shape[0], kernel.shape[1]
    pad_h = kernel_h//2
    pad_w = kernel_w//2

    X_padded = np.pad(X, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output_height = (X.shape[0] + 2 * pad_h - kernel_h) // stride + 1
    output_width = (X.shape[1] + 2 * pad_w - kernel_w) // stride + 1

    result_X = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            values = X_padded[i*stride:i*stride+kernel_h,  j*stride:j*stride+kernel_w]
            result_X[i, j] = np.sum(values*kernel)

    return result_X

if __name__=='__main__':
    image = True
    random_data = True

    if image: 
        # Read image
        X = cv2.imread("512-grayscale-image-Cameraman.png", cv2.IMREAD_GRAYSCALE).astype(int)

        # Apply horizontal edge filter
        kernel_h = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        stride = 2
        result_h_edge = convolution(X, kernel_h, stride)
        
        # Apply vertical edge filter
        kernel_v = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        result_v_edge = convolution(X, kernel_v, stride)

        # Plot the input and output arrays in heatmap form
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

        # Plot input array
        ax1.imshow(X, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')

        # Plot horizontal edge filter output array
        ax2.imshow(result_h_edge, cmap='gray')
        ax2.set_title("Horizontal Edge Image")
        ax2.axis('off')

        # Plot vertical edge filter output array
        ax3.imshow(result_v_edge, cmap='gray')
        ax3.set_title("Vertical Edge Image")
        ax3.axis('off')

        plt.tight_layout()
        plt.savefig("Convolution_image.png")
        plt.show()