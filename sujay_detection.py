import cv2
import numpy as np


# MOST ERRORS CAN BE GOTTEN RID OF VIA SIZE THRESHOLDING USING DISTANCE

# this is to force rescale the image to lower res
def rescale(frame):
    scale_fac = max(frame.shape[0], frame.shape[1]) / 500.0
    if scale_fac > 1:
        return cv2.resize(frame, (int(frame.shape[1] / scale_fac), int(frame.shape[0] / scale_fac)))
    return frame


# Read the original image
img = cv2.imread('Test Images/FloorMix.jpg')
img = rescale(img)

# Create a copy of the original image to draw on
img_dots = img.copy()


def find_detections(image):
    # Canny Edge Detection
    edges = cv2.Canny(image=image, threshold1=100, threshold2=210)  # Canny Edge Detection
    edges = cv2.GaussianBlur(edges, (5, 5), 0)
    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # List to store detected pixels
    detections = []

    # Distance threshold for merging centers
    distance_threshold = 10

    # Mean threshold
    mean_threshold = 0.8

    # Mask to store donuts use to verify donut shape
    inside = 0.5
    outside = 1

    # Iterate over the contours and find insides of the pixels
    for contour in contours:

        # len is a function in python that means length.
        # It returns the number of vertices in the contour.
        if True:  # TODO: eventually add size thresholding based on distance
            # approx) <= 6:
            # bounding rect so far has given the best ellipse estimations (yes i know of fitEllipse)
            x, y, a, b = cv2.boundingRect(contour)
            cx, cy = x + a / 2, y + b / 2

            # ellipse area
            area = a * b * np.pi * inside * inside
            if 1.5 * cv2.contourArea(contour) > area and cy - b >= 0 and cx - a >= 0 and cy + b < image.shape[0] and cx + a < image.shape[1]:
                small_image = image[int(cy - b):int(cy + b), int(cx - a):int(cx + a)]
                mask = np.zeros_like(small_image)
                # writes the donut shape onto the mask
                cv2.ellipse(mask, (int(a), int(b)), (int(outside * a), int(outside * b)), 0, 0.0, 360.0, (1, 1, 1), -1)
                cv2.ellipse(mask, (int(a), int(b)), (int(inside * a), int(inside * b)), 0, 0.0, 360.0, (0, 0, 0), -1)

                active_count = np.sum(mask)
                outer_mean = 0
                if active_count > 0:
                    # BITWISE_AND PER IMAGE IS PRETTY INEFFICIENT, COULD SLOW DOWN BOT
                    # this line finds the mean of the values that are inside the donut written on the mask
                    outer_mean = np.sum(cv2.bitwise_and(small_image, mask)) / active_count

                if outer_mean > mean_threshold:
                    merge = False
                    for det in detections:
                        (det_x, det_y), _ = det
                        # Line below uses the distance formula to check if it is within the distance threshold.
                        if np.sqrt((x - det_x) ** 2 + (y - det_y) ** 2) <= distance_threshold:
                            merge = True
                            break

                    # If it's not close to an existing center, add it to the list and draw the dot
                    if not merge:
                        detections.append(((cx, cy), (a, b)))
                        # cv2.drawContours(edges, [contour], -1, (255, 0, 0), 1)
                        cv2.ellipse(edges, (int(cx), int(cy)), (int(inside * a), int(inside * b)), 0, 0.0, 360.0,
                                    (255, 0, 0), 1)
                        cv2.ellipse(edges, (int(cx), int(cy)), (int(outside * a), int(outside * b)), 0, 0.0, 360.0,
                                    (255, 0, 0), 1)

    cv2.imshow("EDGES", edges)
    cv2.imshow("IMAGE", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return detections


# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# color space conversions
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
# use mean of the image as a baseline for the white mask
# this mean is just temporary, eventually we should find backboard via april tags and sample lighting via that?
mean = np.sum(gray) / (gray.shape[0] * gray.shape[1]) * 1.5
_, white_mask = cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
# these lines ensure that none of the other masks are on the white mask
white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(green_mask))
white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(yellow_mask))
white_mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(purple_mask))
white_detections = find_detections(white_mask)
white_dot = (255, 255, 255)


def draw_detection(detection, color):
    (x, y), (a, b) = detection
    cv2.circle(img_dots, (int(x), int(y)), 2, color, -1)


for detection in yellow_detections:
    draw_detection(detection, yellow_dot)

for detection in green_detections:
    draw_detection(detection, green_dot)

for detection in purple_detections:
    draw_detection(detection, purple_dot)

for detection in white_detections:
    draw_detection(detection, white_dot)

# Display the image with dots
cv2.imshow('Image with Dots', img_dots)

cv2.waitKey(0)
cv2.destroyAllWindows()
