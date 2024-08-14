import cv2
import numpy as np

def preprocess_image(image, target_size):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to target size
    resized = cv2.resize(gray, target_size)
    # Normalize the image
    normalized = resized / 255.0
    return normalized

# Example usage
if __name__ == "__main__":
    sample_image = cv2.imread("screenshot.png")
    processed_image = preprocess_image(sample_image, (84, 84))
    cv2.imwrite("processed_screenshot.png", processed_image)
