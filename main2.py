import cv2
import numpy as np

#contour draw func
def contourDraw(contours, image):
    # image = mainImage.copy()
    for contour in contours:
        # Epsilon defines the accuracy of the approximation.
        # 0.04 is 4% accuracy. The larger the number, the less accurate.

        # The method computes the perimeter of the contour.
        # True is means that the contour is closed.
        epsilon = 0.05 * cv2.arcLength(contour, True)

        # This approximates a contour shape to another shape with fewer vertices based on the epsilon.
        # True indicates that it is a closed contour.
        # Approx returns a list of points (x, y) in each element.
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # len is a function in python that means length.
        # It returns the number of vertices in the contour.
        if cv2.contourArea(contour) > 10 and cv2.contourArea(contour) < 1000000:
            if cv2.isContourConvex(approx): # and 4 <= len(approx) <= 6:
                cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

        # return image


# Read the original image
img = cv2.imread('Test Images/MultiColor.jpg')

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

# White image
w_lower = np.array([0, 0, 140])
w_higher = np.array([179, 60, 255])
white_mask = cv2.inRange(hsv, w_lower, w_higher)

#Canny edgte detection for each one
purple_edges = cv2.Canny(image=purple_mask, threshold1=100, threshold2=210)  # Canny Edge Detection
white_edges = cv2.Canny(image=white_mask, threshold1=100, threshold2=210)  # Canny Edge Detection
green_edges = cv2.Canny(image=green_mask, threshold1=100, threshold2=210)  # Canny Edge Detection
yellow_edges = cv2.Canny(image=yellow_mask, threshold1=100, threshold2=210)  # Canny Edge Detection

#contour
purple_contours, _ = cv2.findContours(purple_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
white_contours, _ = cv2.findContours(white_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
green_contours, _ = cv2.findContours(green_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
yellow_contours, _ = cv2.findContours(yellow_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contourDraw(purple_contours, img)
contourDraw(white_contours, img)
contourDraw(green_contours, img)
contourDraw(yellow_contours, img)

cv2.imshow("main image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()