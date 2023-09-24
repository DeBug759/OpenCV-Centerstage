import cv2
import numpy as np

img = cv2.imread("Test Images/MultiColor.jpg")

print(img.shape)

transformed_image = img.copy()

start = (0, 0)
end = (300, 300)

dim = (640, 641)

bounds = [(50, 50), (440, 0), (530, 575), (170, 650)]


matrix = cv2.getPerspectiveTransform(np.float32(bounds), np.float32([[0, 0], [640, 0], [640, 641], [0, 641]]))
#matrix = cv2.getPerspectiveTransform(np.float32([start, (end[0], start[1]), end, (start[0], end[1])]), np.float32([[0, 0], [640, 0], [640, 641], [0, 641]]))

transformed_image = cv2.warpPerspective(img, matrix, (640, 641))
#print(cv2.getPerspectiveTransform(img, transformed_image))
#cv2.rectangle(img, start, end, (255, 0, 0), 5)
cv2.polylines(img, [np.int32(bounds)], True, (255, 0, 0), 5)

range = cv2.inRange(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2HSV), (0, 0, 170), (360, 30, 255))
cv2.imshow("range", range)

#timg_gray = cv2.cvtColor(range, cv2.COLOR_HSV2GRAY)

contours, _ = cv2.findContours(range, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

boundedcontours = [contour for contour in contours if (cv2.contourArea(contour) < 850 and cv2.contourArea(contour) > 450)]

avg = 0
maxlen = 1000
for contour in boundedcontours:
    avg += cv2.contourArea(contour)
    maxlen = min(maxlen, cv2.contourArea(contour))

# print(avg/len(boundedcontours))
# print(maxlen)
# print(len(boundedcontours))

cv2.drawContours(transformed_image, boundedcontours, -1, (0, 255, 0), 3)



cv2.imshow("thing", transformed_image)
cv2.imshow("thing1", img)

cv2.waitKey(0)

cv2.destroyAllWindows()