import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load video and read the first frame
video_path = r"C:\Users\Sarthak singh\Downloads\P1_example (1).mp4"
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video. Check the path or file.")
    exit()

# Convert to grayscale and apply Canny edge detection
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
canny_edges = cv2.Canny(gray, 50, 150)

# Display side-by-side using matplotlib
plt.figure(figsize=(12, 6))

# Original Frame
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("Original Frame")
plt.axis("off")

# Canny Edge Output
plt.subplot(1, 2, 2)
plt.imshow(canny_edges, cmap='gray')
plt.title("Canny Edge Detection Output")
plt.axis("off")

plt.tight_layout()
plt.savefig("canny_output.png")  # Save as image
plt.show()
