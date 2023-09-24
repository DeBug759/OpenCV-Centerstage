import cv2
import numpy as np

image = cv2.imread('Test Images/test.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), 2)  # You can adjust the kernel size as needed

circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])  # circle center
        cv2.circle(image, center, 1, (0, 100, 100), 3)  # circle center
        radius = i[2]
        cv2.circle(image, center, radius, (255, 0, 255), 2)  # circle outline


cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

