import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def show_image(img, title="Image"):
    plt.figure(figsize=(10, 6))
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# 2. Read and resize
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read image at path: {image_path}")

scale = 800 / image.shape[1]
image = cv2.resize(image, (800, int(image.shape[0] * scale)))
show_image(image, "Original Image")

# 3. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image(gray, "Grayscale Image")

# 4. Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
show_image(blurred, "Blurred Image")

# 5. Adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(
    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 11, 2
)
show_image(adaptive_thresh, "Adaptive Thresholding")

# 6. Morphological closing
kernel = np.ones((7, 7), np.uint8)
closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
show_image(closed, "After Morphological Closing")

# 7. Contour detection & filtering
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_img = image.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 4000:  # Remove small false positives
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 3:  # Ignore thin cracks
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

show_image(output_img, "Final Pothole Detection")
