import cv2
import numpy as np

image = cv2.imread('Test Images/test.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 6:
        # Check if the detected hexagon is convex. This is optional but can help in filtering out false positives.
        if cv2.isContourConvex(approx):
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

cv2.imshow('Detected Hexagons', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
