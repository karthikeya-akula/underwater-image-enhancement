import cv2

def load_image(path):
    """
    Load an image from disk.
    """
    image = cv2.imread(path)
    return image

def resize_image(image, size=(512, 512)):
    """
    Resize image to a fixed size.
    """
    resized = cv2.resize(image, size)
    return resized

def save_image(path, image):
    """
    Save image to disk.
    """
    cv2.imwrite(path, image)