import cv2
import numpy as np

# Read the original image
img = cv2.imread('Test Images/test.png')

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (13, 13), 2)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=210)  # Canny Edge Detection
# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Create a copy of the original image to draw on
img_dots = img.copy()

# Set the color of the dots to green (BGR format)
dot_color = (245, 66, 236)

# List to store detected centers
centers = []

# Lists to store the x and y coordinates of hexagons
all_x = []
all_y = []

# Iterate over the contours and find those with approximately 6 vertices (hexagons)
for contour in contours:
    # Epsilon defines the accuracy of the approximation.
    # 0.04 is 4% accuracy. The larger the number, the less accurate.

    # The method computes the perimeter of the contour.
    # True is means that the contour is closed.
    epsilon = 0.04 * cv2.arcLength(contour, True)

    # This approximates a contour shape to another shape with fewer vertices based on the epsilon.
    # True indicates that it is a closed contour.
    # Approx returns a list of points (x, y) in each element.
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # len is a function in python that means length.
    # It returns the number of vertices in the contour.
    if len(approx) == 6:
        for point in approx:
            all_x.append(point[0][0])
            all_y.append(point[0][1])

# Find the bounding box that encompasses all the hexagons
x_min, x_max = min(all_x), max(all_x)
y_min, y_max = min(all_y), max(all_y)

# Crop the image to the bounding box
cropped_img = img_dots[y_min:y_max, x_min:x_max]

# Distance threshold for merging centers
distance_threshold = 10

# Calculate center
for contour in contours:
    if len(contour) > 0:
        # Calculate the center of the hexagon
        # cX = int(M["m10"] / M["m00"])
        # m10 are a sum of the moments (mathematical term that defines a shape of a graph) related to the x-axis.
        # m01 are the moments that are related to the y-axis.
        M = cv2.moments(contour)
        if M["m00"] != 0:
            # Dividing the values gives us the center in the x and y.
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Check if the center is close to an existing center
            merge = False
            for center in centers:
                # Line below uses the distance formula to check if it is within the distance threshold.
                if np.sqrt((cX - center[0]) ** 2 + (cY - center[1]) ** 2) <= distance_threshold:
                    merge = True
                    break

            # If it's not close to an existing center, add it to the list and draw the dot
            if not merge:
                centers.append((cX, cY))
                cv2.circle(img_dots, (cX, cY), 5, dot_color, -1)  # -1 fills the circle

# Sort centers by y-coordinate and then by x-coordinate
sorted_centers = sorted(centers, key=lambda pt: (pt[1], pt[0]))

# Define a threshold for grid cell size
x_threshold = (x_max - x_min) // 6  # Assuming 6 hexagons horizontally
y_threshold = (y_max - y_min) // 6  # Assuming 6 hexagons vertically

# Display the image with green dots
cv2.imshow('Image with Dots', img_dots)
cv2.imshow('Edges', edges)
print(centers)

cv2.waitKey(0)
cv2.destroyAllWindows()