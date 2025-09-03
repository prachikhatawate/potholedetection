# Pothole Detection using Adaptive Thresholding & Texture Analysis

## 📌 Problem Statement
Detect potholes on road images by analyzing **texture changes** with filters and **adaptive thresholding** techniques.

## 🎯 Objectives
- Detect potholes on road surfaces.
- Use classical Computer Vision (CV) instead of deep learning.
- Highlight potholes using bounding boxes.

## 🛠️ Core Techniques
- Image Preprocessing (Grayscale, Gaussian Blur)
- Edge Detection & Adaptive Thresholding
- Morphological Operations
- Contour Detection & Filtering

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/pothole-detection-adaptive-thresholding.git
   cd pothole-detection-adaptive-thresholding
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script (Google Colab supported):
   ```bash
   python pothole_detection.py
   ```

4. Upload an image when prompted.  
   The program will:
   - Show original, grayscale, blurred, thresholded, and final detected potholes.

## 📷 Example Output
Detected potholes are highlighted with **green bounding boxes**.

## ⚙️ Requirements
- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Google Colab (if running online)

---
✍️ Developed as part of a project on **Road Surface Pothole Detection** using Adaptive Thresholding and Texture Analysis.
