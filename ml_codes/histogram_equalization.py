def histogram_equalizer(img1, img2=None):
    """
    Apply histogram equalization to grayscale images.
    
    Parameters:
    -----------
    img1 : numpy.ndarray
        Input grayscale image to be equalized
    img2 : numpy.ndarray, optional
        If provided, img1's histogram will be matched to img2's histogram
        
    Returns:
    --------
    numpy.ndarray
        Equalized image
    """
    import numpy as np
    
    # Make a copy of the image to avoid modifying the original
    img_result = img1.copy()
    
    # Get image dimensions
    height, width = img1.shape[:2]
    
    if img2 is None:
        # Simple histogram equalization
        
        # Step 1: Calculate histogram
        histogram = np.zeros(256, dtype=int)
        for i in range(height):
            for j in range(width):
                pixel_value = img1[i, j]
                histogram[pixel_value] += 1
        
        # Step 2: Calculate cumulative distribution function (CDF)
        cdf = np.zeros(256, dtype=float)
        cdf[0] = histogram[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + histogram[i]
        
        # Step 3: Normalize CDF to [0, 255]
        cdf_min = cdf[np.nonzero(cdf)[0][0]] if np.any(cdf) else 0
        cdf_normalized = np.round(((cdf - cdf_min) / (height * width - cdf_min)) * 255)
        
        # Step 4: Map each pixel value to its equalized value
        for i in range(height):
            for j in range(width):
                img_result[i, j] = cdf_normalized[img1[i, j]]
    
    else:
        # Histogram matching (specification)
        
        # Calculate histogram and CDF for source image (img1)
        hist_src = np.zeros(256, dtype=int)
        for i in range(height):
            for j in range(width):
                hist_src[img1[i, j]] += 1
        
        cdf_src = np.zeros(256, dtype=float)
        cdf_src[0] = hist_src[0]
        for i in range(1, 256):
            cdf_src[i] = cdf_src[i-1] + hist_src[i]
        
        # Normalize source CDF
        cdf_src_min = cdf_src[np.nonzero(cdf_src)[0][0]] if np.any(cdf_src) else 0
        cdf_src_normalized = ((cdf_src - cdf_src_min) / (height * width - cdf_src_min))
        
        # Calculate histogram and CDF for reference image (img2)
        h_ref, w_ref = img2.shape[:2]
        hist_ref = np.zeros(256, dtype=int)
        for i in range(h_ref):
            for j in range(w_ref):
                hist_ref[img2[i, j]] += 1
        
        cdf_ref = np.zeros(256, dtype=float)
        cdf_ref[0] = hist_ref[0]
        for i in range(1, 256):
            cdf_ref[i] = cdf_ref[i-1] + hist_ref[i]
        
        # Normalize reference CDF
        cdf_ref_min = cdf_ref[np.nonzero(cdf_ref)[0][0]] if np.any(cdf_ref) else 0
        cdf_ref_normalized = ((cdf_ref - cdf_ref_min) / (h_ref * w_ref - cdf_ref_min))
        
        # Create mapping function using CDFs
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            if cdf_src_normalized[i] > 0:
                j = 0
                while j < 256 and cdf_ref_normalized[j] < cdf_src_normalized[i]:
                    j += 1
                mapping[i] = j
        
        # Apply mapping to source image
        for i in range(height):
            for j in range(width):
                img_result[i, j] = mapping[img1[i, j]]
    
    return img_result

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    
    # Read images
    poor_contrast = cv2.imread("poor_contrast.png", cv2.IMREAD_GRAYSCALE)
    cameraman = cv2.imread("512-grayscale-image-Cameraman.png", cv2.IMREAD_GRAYSCALE)
    
    if poor_contrast is None or cameraman is None:
        print("Error: Could not read one or both images")
        print("Make sure both images are in the correct directory")
        exit()
    
    # Apply histogram equalization
    equalized_image = histogram_equalizer(cameraman)
    matched_image = histogram_equalizer( cameraman, poor_contrast)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original poor contrast image
    axes[0, 0].imshow(poor_contrast, cmap='gray')
    axes[0, 0].set_title('Original Poor Contrast Image')
    axes[0, 0].axis('off')
    
    # Reference cameraman image
    axes[0, 1].imshow(cameraman, cmap='gray')
    axes[0, 1].set_title('Reference Cameraman Image')
    axes[0, 1].axis('off')
    
    # Standard histogram equalization
    axes[1, 0].imshow(equalized_image, cmap='gray')
    axes[1, 0].set_title('After Histogram Equalization')
    axes[1, 0].axis('off')
    
    # Histogram matched image
    axes[1, 1].imshow(matched_image, cmap='gray')
    axes[1, 1].set_title('After Histogram Matching with Cameraman')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('histogram_equalization_result.png')
    plt.show()
    
    # Plot histograms for comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot histogram of poor contrast image
    axes[0, 0].hist(poor_contrast.flatten(), 256, [0, 256], color='gray')
    axes[0, 0].set_title('Original Poor Contrast Histogram')
    axes[0, 0].set_xlim([0, 256])
    
    # Plot histogram of cameraman image
    axes[0, 1].hist(cameraman.flatten(), 256, [0, 256], color='gray')
    axes[0, 1].set_title('Cameraman Histogram')
    axes[0, 1].set_xlim([0, 256])
    
    # Plot histogram of equalized image
    axes[1, 0].hist(equalized_image.flatten(), 256, [0, 256], color='gray')
    axes[1, 0].set_title('Equalized Histogram')
    axes[1, 0].set_xlim([0, 256])
    
    # Plot histogram of matched image
    axes[1, 1].hist(matched_image.flatten(), 256, [0, 256], color='gray')
    axes[1, 1].set_title('Matched Histogram')
    axes[1, 1].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig('histogram_comparison.png')
    plt.show()
