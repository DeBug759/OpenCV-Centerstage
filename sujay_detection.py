import cv2
import numpy as np


# this is to force rescale the image to display on my computer
def rescale(frame):
    scale_fac = max(frame.shape[0], frame.shape[1]) / 500.0
    if scale_fac > 1:
        return cv2.resize(frame, (int(frame.shape[1] / scale_fac), int(frame.shape[0] / scale_fac)))
    return frame


# this is to display an image on to my computer
def display(name, frame):
    cv2.imshow(name, rescale(frame))


# Read the original image
img = cv2.imread('Test Images/test2.JPG')
img = rescale(img)

# Create a copy of the original image to draw on
img_dots = img.copy()


def find_detections(image):
    # Canny Edge Detection
    edges = cv2.Canny(image=image, threshold1=100, threshold2=210)  # Canny Edge Detection
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    # edges = cv2.bitwise_not(image)
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # List to store detected pixels
    detections = []

    # Distance threshold for merging centers
    distance_threshold = 10
    # Iterate over the contours and find those with approximately 6 vertices (hexagons)
    for contour in contours:

        # len is a function in python that means length.
        # It returns the number of vertices in the contour.
        if (image.shape[0] / 10.0 * image.shape[1] / 10.0) > cv2.contourArea(contour) > (
                image.shape[0] / 100.0 * image.shape[1] / 100.0):  # and cv2.contourArea(contour) < 20: 4 <= len(
            # approx) <= 6:
            (x, y), (a, b), ang = cv2.minAreaRect(contour)

            mask = np.zeros_like(image)
            cv2.ellipse(mask, (int(x), int(y)), (int(a), int(b)), ang, 0.0, 360.0, (1, 1, 1), -1)
            cv2.ellipse(mask, (int(x), int(y)), (int(0.4 * a), int(0.4 * b)), ang, 0.0, 360.0, (0, 0, 0), -1)

            mean = np.sum(np.multiply(image / 255.0, mask)) / np.sum(mask)

            area = a * b * np.pi / 4
            if area < cv2.contourArea(contour) * 1.05 and mean > 0.75:
                merge = False
                for det in detections:
                    (det_x, det_y), _ = det
                    # Line below uses the distance formula to check if it is within the distance threshold.
                    if np.sqrt((x - det_x) ** 2 + (y - det_y) ** 2) <= distance_threshold:
                        merge = True
                        break

                # If it's not close to an existing center, add it to the list and draw the dot
                if not merge:
                    detections.append(((x, y), area))
                    cv2.drawContours(edges, [contour], -1, (255, 0, 0), 1)
                    cv2.ellipse(edges, (int(x), int(y)), (int(a), int(b)), ang, 0.0, 360.0, (255, 0, 0), 1)
    display("EDGES", edges)
    display("IMAGE", image)
    cv2.waitKey()
    return detections


# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# HSV conversion
hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

# yellow
y_lower = np.array([0, 102, 93])
y_higher = np.array([35, 255, 255])
yellow_mask = cv2.inRange(hsv, y_lower, y_higher)
yellow_detections = find_detections(yellow_mask)
yellow_dot = (0, 255, 255)

# green
g_lower = np.array([34, 40, 98])
g_higher = np.array([73, 255, 255])
green_mask = cv2.inRange(hsv, g_lower, g_higher)
green_detections = find_detections(green_mask)
green_dot = (0, 128, 0)

# purple
p_lower = np.array([117, 25, 92])
p_higher = np.array([172, 255, 255])
purple_mask = cv2.inRange(hsv, p_lower, p_higher)
purple_detections = find_detections(purple_mask)
purple_dot = (128, 0, 128)

# white
w_lower = np.array([0, 0, 140])
w_higher = np.array([179, 24, 255])
mean = np.sum(gray) / (gray.shape[0] * gray.shape[1]) * 1.5
_, white_mask = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(green_mask))
white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(yellow_mask))
white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(purple_mask))
white_detections = find_detections(white_mask)
white_dot = (255, 255, 255)

# Set the thickness to draw everything at
thickness = int(np.ceil(img.shape[0] / 200.0))
print(thickness)

for detection in yellow_detections:
    (x, y), _ = detection
    cv2.circle(img_dots, (int(x), int(y)), thickness, yellow_dot, -1)

for detection in green_detections:
    (x, y), _ = detection
    cv2.circle(img_dots, (int(x), int(y)), thickness, green_dot, -1)

for detection in purple_detections:
    (x, y), _ = detection
    cv2.circle(img_dots, (int(x), int(y)), thickness, purple_dot, -1)

for detection in white_detections:
    (x, y), _ = detection
    cv2.circle(img_dots, (int(x), int(y)), thickness, white_dot, -1)

# Display the image with dots
display('White Mask', white_mask)
display('Image with Dots', img_dots)

cv2.waitKey(0)
cv2.destroyAllWindows()
