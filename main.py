import os
import cv2
import numpy as np

from src.pipeline import run_pipeline

from src.preprocessing import (
    load_image,
    resize_image,
    save_image,
)

from src.enhancement import (
    histogram_equalization,
    clahe_enhancement,
    gamma_correction,
    white_balance,
)

from src.logger import logger


# Run pipeline
run_pipeline()


RAW_DIR = "data/raw"

OUTPUT_DIRS = {
    "histogram_eq": "outputs/histogram_eq",
    "clahe": "outputs/clahe",
    "gamma": "outputs/gamma",
    "white_balance": "outputs/white_balance",
}

COMPARISON_DIR = "outputs/comparison_visuals"


# Create output directories
for folder in OUTPUT_DIRS.values():
    os.makedirs(folder, exist_ok=True)

os.makedirs(COMPARISON_DIR, exist_ok=True)


logger.info("Enhancement pipeline started")


# Process all images
for filename in os.listdir(RAW_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        input_path = os.path.join(RAW_DIR, filename)

        # Load image
        image = load_image(input_path)

        # Resize image
        image = resize_image(image)

        # Apply enhancement methods
        hist_img = histogram_equalization(image)

        clahe_img = clahe_enhancement(image)

        gamma_img = gamma_correction(
            image,
            gamma=0.7
        )

        wb_img = white_balance(image)

        # Save outputs
        save_image(
            os.path.join(
                OUTPUT_DIRS["histogram_eq"],
                filename
            ),
            hist_img
        )

        save_image(
            os.path.join(
                OUTPUT_DIRS["clahe"],
                filename
            ),
            clahe_img
        )

        save_image(
            os.path.join(
                OUTPUT_DIRS["gamma"],
                filename
            ),
            gamma_img
        )

        save_image(
            os.path.join(
                OUTPUT_DIRS["white_balance"],
                filename
            ),
            wb_img
        )

        print(f"Processed: {filename}")

        logger.info(f"Processed {filename}")


        # -------------------------------
        # Create comparison visualization
        # -------------------------------

        original = cv2.resize(image, (300, 300))

        hist_img = cv2.resize(hist_img, (300, 300))
        clahe_img = cv2.resize(clahe_img, (300, 300))
        gamma_img = cv2.resize(gamma_img, (300, 300))
        wb_img = cv2.resize(wb_img, (300, 300))

        images = [
            original,
            hist_img,
            clahe_img,
            gamma_img,
            wb_img
        ]

        labels = [
            "Original",
            "Histogram EQ",
            "CLAHE",
            "Gamma",
            "White Balance"
        ]

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

        comparison = np.hstack(labeled_images)

        comparison_path = os.path.join(
            COMPARISON_DIR,
            filename
        )

        cv2.imwrite(comparison_path, comparison)

        print(f"Comparison created: {filename}")


logger.info("Enhancement pipeline completed")

print("All enhancement outputs generated successfully!")
print("All comparison visuals created successfully!")