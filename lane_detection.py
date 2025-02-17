import cv2
import numpy as np
from ultralytics import YOLO


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    else:
        left_line = None
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    else:
        right_line = None
    return np.array([line for line in [left_line, right_line] if line is not None])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


# Load YOLO model
model = YOLO('yolo11n.pt')

# Process the video
cap = cv2.VideoCapture(r"C:\Users\Sarthak singh\Downloads\P1_example.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video processing completed or failed to read the frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    canny = cv2.Canny(gray, 50, 150)

    # Mask the region of interest
    cropped_image = region_of_interest(canny)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    # Calculate averaged lines
    averaged_lines = average_slope_intercept(frame, lines)

    # Draw the lines
    line_image = display_lines(frame, averaged_lines)

    # Blend with the original frame
    lane_result = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # Perform YOLO inference
    results = model.predict(source=frame, verbose=False)

    # Draw YOLO detections
    for box in results[0].boxes.xyxy:  # xyxy format
        x1, y1, x2, y2 = map(int, box[:4])
        label = results[0].names[int(results[0].boxes.cls[0])]
        cv2.rectangle(lane_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(lane_result, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the combined result
    cv2.imshow("Result", lane_result)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
