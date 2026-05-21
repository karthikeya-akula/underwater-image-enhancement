import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


def calculate_psnr(original, enhanced):
    return peak_signal_noise_ratio(original, enhanced)


def calculate_ssim(original, enhanced):
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    score, _ = structural_similarity(
        gray_original,
        gray_enhanced,
        full=True
    )

    return score


def calculate_uiqm(image):
    """
    Simplified UIQM approximation.
    """

    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    # Contrast
    contrast = np.std(l)

    # Sharpness
    laplacian = cv2.Laplacian(l, cv2.CV_64F)
    sharpness = laplacian.var()

    # Colorfulness
    colorfulness = np.std(a) + np.std(b)

    uiqm = (
        0.0282 * colorfulness
        + 0.2953 * sharpness
        + 3.5753 * contrast
    )

    return uiqm