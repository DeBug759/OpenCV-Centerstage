import cv2
import numpy as np

# Load the image
image = cv2.imread("SpikeImages/BlueRight.jpg")

# Get the image dimensions
height, width, _ = image.shape

# Calculate the coordinates for the bottom left corner
bottom_left_x = 0
bottom_left_y = height

# Calculate the coordinates for the center
center_x = width // 2
center_y = height // 2

# Define the width and height of the bounding boxes
box_width = 100
box_height = 10

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_hsv = np.array([50, 116, 0])
higher_hsv = np.array([179, 255, 147])
mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

# Draw a bounding box in the bottom left
cv2.rectangle(image, (0, 300), (130, 480), (0, 255, 0), 2)
cv2.rectangle(mask, (0, 300), (130, 480), (255, 255, 255), 2)

# Draw a bounding box in the center
cv2.rectangle(image, (300, 270), (430, 430), (0, 0, 255), 2)
cv2.rectangle(mask, (300, 270), (430, 430), (255, 255, 255), 2)

# Count white pixels in the left bounding box
left_box = mask[300:480, 0:130]
left_white_pixel_count = cv2.countNonZero(left_box)

# Count white pixels in the center bounding box
center_box = mask[270:430, 300:430]
center_white_pixel_count = cv2.countNonZero(center_box)

# Determine the condition and print accordingly
if left_white_pixel_count > 3000:
    print(1)  # A large quantity of white in the left bounding box
elif center_white_pixel_count > 3000:
    print(2)  # A lot of white in the center bounding box
else:
    print(3)  # Neither of the conditions met

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
cv2.imshow('Mask Image', mask)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
