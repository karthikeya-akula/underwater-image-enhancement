import os
import cv2
import numpy as np

from src.preprocessing import load_image

RAW_DIR = "data/raw"

METHODS = {
    "Histogram EQ": "outputs/histogram_eq",
    "CLAHE": "outputs/clahe",
    "Gamma": "outputs/gamma",
    "White Balance": "outputs/white_balance",
}

OUTPUT_DIR = "outputs/comparison_visuals"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        # Load original image
        original_path = os.path.join(RAW_DIR, filename)
        original = load_image(original_path)

        original = cv2.resize(original, (300, 300))

        images = [original]
        labels = ["Original"]

        # Load enhanced images
        for method_name, folder in METHODS.items():

            path = os.path.join(folder, filename)

            img = load_image(path)

            img = cv2.resize(img, (300, 300))

            images.append(img)
            labels.append(method_name)

        # Add labels to images
        labeled_images = []

        for img, label in zip(images, labels):

            img_copy = img.copy()

            cv2.putText(
                img_copy,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            labeled_images.append(img_copy)

        # Concatenate images horizontally
        comparison = np.hstack(labeled_images)

        # Save output
        output_path = os.path.join(OUTPUT_DIR, filename)

        cv2.imwrite(output_path, comparison)

        print(f"Comparison created: {filename}")

print("All comparison visuals created successfully!")