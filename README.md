# Image Segmentation with K-means

A simple Python implementation of image segmentation using K-means clustering. This project reduces the color palette of images while preserving their main features.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python src/image_segmentation.py --image examples/butterfly.png --clusters 4
```

## Features

- Image segmentation using K-means clustering
- Adjustable number of color clusters
- Side-by-side comparison of results
- Automatic saving of segmented images

## Algorithm

The image segmentation process:
1. Convert image to RGB and reshape pixels into a 2D array
2. Apply K-means clustering to group similar colors
3. Replace each pixel with its cluster center color
4. Reconstruct and save the segmented image

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
