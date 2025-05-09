import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import argparse

def load_image(image_path):
    """
    Load and return the image from the given path.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img

def segment_image(img, n_clusters=3):
    """
    Perform K-means clustering on the image.
    
    Args:
        img (numpy.ndarray): Input image
        n_clusters (int): Number of clusters for K-means
        
    Returns:
        numpy.ndarray: Segmented image
    """
    # Reshape the image to be a list of pixels
    h, w, d = img.shape
    image_array = img.reshape((h * w, d))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(image_array)
    
    # Create the segmented image
    segmented_image = kmeans.cluster_centers_[labels].reshape(h, w, d)
    return segmented_image.astype(np.uint8)

def display_results(original_img, segmented_img):
    """
    Display the original and segmented images side by side.
    
    Args:
        original_img (numpy.ndarray): Original image
        segmented_img (numpy.ndarray): Segmented image
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Image Segmentation using K-means Clustering')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--clusters', type=int, default=3, help='Number of clusters for segmentation')
    args = parser.parse_args()
    
    # Load and process the image
    img = load_image(args.image)
    segmented_img = segment_image(img, args.clusters)
    
    # Display results
    display_results(img, segmented_img)
    
    # Save the segmented image
    output_path = 'segmented_' + args.image.split('/')[-1]
    cv2.imwrite(output_path, segmented_img)
    print(f"Segmented image saved as: {output_path}")

if __name__ == "__main__":
    main() 