import cv2
import numpy as np

# Read the original image
img = cv2.imread('Test Images/test2.JPG')

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img.copy(), (5, 5), 0)

# HSV conversion
hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

# yellow
y_lower = np.array([0, 102, 93])
y_higher = np.array([35, 255, 255])
yellow_mask = cv2.inRange(hsv, y_lower, y_higher)

# green
g_lower = np.array([34, 62, 48])
g_higher = np.array([73, 255, 255])
green_mask = cv2.inRange(hsv, g_lower, g_higher)

# purple
p_lower = np.array([117, 25, 92])
p_higher = np.array([172, 255, 255])
purple_mask = cv2.inRange(hsv, p_lower, p_higher)

# white
w_lower = np.array([0, 0, 140])
w_higher = np.array([179, 60, 255])
white_mask = cv2.inRange(hsv, w_lower, w_higher)

greenYellow = cv2.bitwise_or(yellow_mask, green_mask)
purpleWhite= cv2.bitwise_or(purple_mask, white_mask)

res = cv2.bitwise_or(greenYellow, purpleWhite)

# Canny Edge Detection
edges = cv2.Canny(image=res, threshold1=100, threshold2=210)  # Canny Edge Detection
edges = cv2.GaussianBlur(edges, (5, 5), 0)
# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Create a copy of the original image to draw on
img_dots = img.copy()

# Set the color of the dots to green (BGR format)
dot_color = (245, 66, 236)

# List to store detected centers
centers = []

# Distance threshold for merging centers
distance_threshold = 10

# Iterate over the contours and find those with approximately 6 vertices (hexagons)
for contour in contours:
    # Epsilon defines the accuracy of the approximation.
    # 0.04 is 4% accuracy. The larger the number, the less accurate.

    # The method computes the perimeter of the contour.
    # True is means that the contour is closed.
    epsilon = 0.02 * cv2.arcLength(contour, True)

    # This approximates a contour shape to another shape with fewer vertices based on the epsilon.
    # True indicates that it is a closed contour.
    # Approx returns a list of points (x, y) in each element.
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # len is a function in python that means length.
    # It returns the number of vertices in the contour.
    if cv2.contourArea(contour) > (img_blur.shape[0] / 100.0 * img_blur.shape[1] / 100.0): #and cv2.contourArea(contour) < 20: 4 <= len(approx) <= 6:
        (x, y), (a, b), ang = cv2.fitEllipse(contour)
        area = a * b * np.pi / 4
        if area < cv2.contourArea(approx) * 1.1:

            merge = False
            for center in centers:
                # Line below uses the distance formula to check if it is within the distance threshold.
                if np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= distance_threshold:
                    merge = True
                    break

            # If it's not close to an existing center, add it to the list and draw the dot
            if not merge:
                centers.append((x, y))
                cv2.circle(img_dots, (int(x), int(y)), 3, dot_color, -1)  # -1 fills the circle
                cv2.drawContours(img_dots, [approx], 0, (0, 255, 0), 2)

        # for point in approx:
        #     all_x.append(point[0][0])
        #     all_y.append(point[0][1])

#
# # Calculate center
# for contour in contours:
#     if len(contour) > 0:
#         # Calculate the center of the hexagon
#         # cX = int(M["m10"] / M["m00"])
#         # m10 are a sum of the moments (mathematical term that defines a shape of a graph) related to the x-axis.
#         # m01 are the moments that are related to the y-axis.
#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             # Dividing the values gives us the center in the x and y.
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#
#             # Check if the center is close to an existing center
#             merge = False
#             for center in centers:
#                 # Line below uses the distance formula to check if it is within the distance threshold.
#                 if np.sqrt((cX - center[0]) ** 2 + (cY - center[1]) ** 2) <= distance_threshold:
#                     merge = True
#                     break
#
#             # If it's not close to an existing center, add it to the list and draw the dot
#             if not merge:
#                 centers.append((cX, cY))
#                 cv2.circle(img_dots, (cX, cY), 3, dot_color, -1)  # -1 fills the circle
#
# # Sort centers by y-coordinate and then by x-coordinate
# sorted_centers = sorted(centers, key=lambda pt: (pt[1], pt[0]))

# Define a threshold for grid cell size
#x_threshold = (x_max - x_min) // 6  # Assuming 6 hexagons horizontally
#y_threshold = (y_max - y_min) // 6  # Assuming 6 hexagons vertically

#Display the image with green dots
cv2.imshow('Edges', edges)
cv2.imshow('res', res)
cv2.imshow('Image with Dots', img_dots)

print(centers)

cv2.waitKey(0)
cv2.destroyAllWindows()