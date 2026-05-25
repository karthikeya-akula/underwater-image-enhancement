import os

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


# Create output directories
for folder in OUTPUT_DIRS.values():
    os.makedirs(folder, exist_ok=True)


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


logger.info("Enhancement pipeline completed")

print("All enhancement outputs generated successfully!")