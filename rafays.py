import cv2
import matplotlib.pyplot as plt
from fast_plate_ocr import ONNXPlateRecognizer

# Path to the image
image_path = "image.png"

# Load Fast Plate OCR model
plate_recognizer = ONNXPlateRecognizer('argentinian-plates-cnn-model')

# Read image
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Error: Image not found. Check the file path.")

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Now shape is (H, W)

# Apply OCR
plate_texts = plate_recognizer.run(img_gray)  # Pass grayscale image

detected_text = "\n".join(plate_texts) if plate_texts else "No Plate Detected"

# Display image
plt.figure(figsize=(6, 6))
plt.imshow(img_gray, cmap="gray")
plt.axis("off")
plt.title(detected_text, fontsize=10)
plt.show()
