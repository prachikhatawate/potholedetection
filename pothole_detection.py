import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern

# ----------------------------
# Preprocessing + Mask Creation
# ----------------------------
def process_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Downscale for noise reduction
    h, w = gray.shape
    scale = 0.5
    gray = cv2.resize(gray, (int(w*scale), int(h*scale)))

    # Stronger blur
    blur = cv2.medianBlur(gray, 5)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 3
    )

    # LBP texture
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    _, lbp_mask = cv2.threshold(lbp_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine
    combined = cv2.bitwise_and(thresh, lbp_mask)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    return img, combined, scale

# ----------------------------
# Merge Overlapping Bounding Boxes
# ----------------------------
def merge_boxes(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    areas = boxes[:,2] * boxes[:,3]
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / areas[idxs[:-1]]

        idxs = np.delete(
            idxs,
            np.concatenate(([len(idxs) - 1],
                           np.where(overlap > overlapThresh)[0]))
        )

    return boxes[pick].astype("int")

# ----------------------------
# Draw bounding boxes
# ----------------------------
def draw_boxes(img, combined, scale):
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contours before filtering: {len(contours)}")

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area > 2000:  # keep only bigger regions
            # Scale back to original
            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            boxes.append([x, y, w, h])

    # Merge overlapping boxes
    merged_boxes = merge_boxes(boxes, overlapThresh=0.2)

    for (x, y, w, h) in merged_boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"Boxes after merging: {len(merged_boxes)}")
    return img

# ----------------------------
# Run on all dataset images
# ----------------------------
def run_on_dataset(dataset_path, output_path, show=True):
    os.makedirs(output_path, exist_ok=True)
    all_images = [f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    for img_name in all_images:
        image_path = os.path.join(dataset_path, img_name)
        img, combined, scale = process_image(image_path)
        img_with_boxes = draw_boxes(img.copy(), combined, scale)

        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        side_by_side = np.hstack((img, img_with_boxes, cv2.resize(combined_bgr, (img.shape[1], img.shape[0]))))

        save_path = os.path.join(output_path, img_name)
        cv2.imwrite(save_path, side_by_side)

        if show:
            cv2.imshow("Original | Detection | Mask", side_by_side)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC to quit
                break

        print(f"Processed and saved: {save_path}")

    cv2.destroyAllWindows()

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    dataset_path = r"D:\project\test"
    output_path = r"D:\project\output"
    run_on_dataset(dataset_path, output_path, show=True)
