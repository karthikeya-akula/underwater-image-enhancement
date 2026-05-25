import cv2
import numpy as np


def histogram_equalization(image):
    """
    Apply Histogram Equalization
    """

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(ycrcb)

    y_eq = cv2.equalizeHist(y)

    merged = cv2.merge((y_eq, cr, cb))

    result = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    return result


def clahe_enhancement(image):
    """
    Apply CLAHE enhancement
    """

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_clahe = clahe.apply(l)

    merged = cv2.merge((l_clahe, a, b))

    result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return result


def gamma_correction(image, gamma=0.7):
    """
    Apply Gamma Correction
    """

    table = np.array([
        ((i / 255.0) ** gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    corrected = cv2.LUT(image, table)

    return corrected


def white_balance(image):
    """
    Apply White Balance Correction
    """

    img = image.astype(np.float32)

    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    img[:, :, 0] *= avg_gray / avg_b
    img[:, :, 1] *= avg_gray / avg_g
    img[:, :, 2] *= avg_gray / avg_r

    img = np.clip(img, 0, 255)

    return img.astype(np.uint8)