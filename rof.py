import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Load the first frame from the video
video_path = r"C:\Users\Sarthak singh\Downloads\P1_example (1).mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video frame.")
    exit()

# Convert to grayscale and apply Canny Edge Detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 150)

# Apply ROI mask
roi_result = region_of_interest(canny)

# Display side-by-side
plt.figure(figsize=(12, 6))

# Canny Edges
plt.subplot(1, 2, 1)
plt.imshow(canny, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")

# ROI Masked Output
plt.subplot(1, 2, 2)
plt.imshow(roi_result, cmap='gray')
plt.title("Region of Interest Masked")
plt.axis("off")

plt.tight_layout()
plt.savefig("roi_output.png")  # Save as image
plt.show()
