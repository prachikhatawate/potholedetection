# Pothole Detection using Adaptive Thresholding & Texture Analysis

## Problem Statement
Detect potholes on road images by analyzing **texture changes** with filters and **adaptive thresholding** techniques.

## Objectives
- Detect potholes on road surfaces.
- Use classical Computer Vision (CV) instead of deep learning.
- Highlight potholes using bounding boxes.

## Core Techniques
- Image Preprocessing (Grayscale, Gaussian Blur)
- Edge Detection & Adaptive Thresholding
- Morphological Operations
- Contour Detection & Filtering

## Dataset Description

This dataset consists of a combination of road images sourced from Google and an existing pothole dataset available on Kaggle. It is organized into three directories: train, val, and test.
-The train folder contains 1,167 images
-The validation (val) folder contains 108 images
-The test folder contains 136 images<br>
-Each image is labeled as either "Normal" (no pothole) or "Pothole" based on its content.

## Algorithm/system design or architecture
![Uploading image.png…]()

## Methodology
-Convert input road image to grayscale.
-Apply Gaussian blur to reduce noise.
-Perform texture analysis using Local Binary Pattern (LBP).
-Apply adaptive thresholding to highlight pothole regions.
-Combine LBP and threshold masks.
-Refine results using morphological operations.
-Detect contours and draw bounding boxes around potholes.

## Code overview
The project is implemented in Python using OpenCV and scikit-image.
Main steps in the code:
process_image() → Preprocess image, apply Gaussian blur, LBP, and adaptive thresholding to generate masks.
merge_boxes() → Merge overlapping bounding boxes for cleaner detection.
draw_boxes() → Detect contours, filter by area, and draw bounding boxes around potholes.
run_on_dataset() → Run detection on all images in the dataset, save outputs, and optionally display results.
Input: Road images from dataset (.jpg).
Output: Processed images with potholes highlighted by green bounding boxes.

## Results 
The system successfully detects potholes on road surface images using texture analysis (LBP) and adaptive thresholding.
Detected pothole regions are highlighted with green bounding boxes.
Noise and small regions are removed using morphological operations.

## Conclusion
This project demonstrates a classical computer vision approach for pothole detection without relying on deep learning. 
By combining texture analysis (LBP) with adaptive thresholding and morphological operations, the system can successfully highlight pothole regions in road surface images.
The method is:
Lightweight (runs on CPU without GPU).
Dataset independent (works on different road conditions with minimal changes).
Explainable (each step in the pipeline is transparent and easy to understand).
Future improvements can include:
Fine-tuning contour filtering for higher accuracy.
Integrating with real-time video streams for live pothole detection.
Using hybrid approaches combining CV with machine learning for better robustness.
