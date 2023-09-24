import cv2
import numpy as np

def putText(mat,text,r,g,b,x,y):
    cv2.putText(mat, text, (x, y), cv2.QT_FONT_NORMAL,
                0.7, (r, g, b), 1, cv2.LINE_AA)


# HSV ARRAYS
lower_white = np.array([0,0,200])
upper_white = np.array([40,80,255])

lower_yellow = np.array([0, 150, 0])
upper_yellow = np.array([45, 255, 255])

lower_purple = np.array([125, 50, 150])
upper_purple = np.array([180, 100, 255])

lower_green = np.array([50, 130, 0])
upper_green = np.array([80, 255, 255])

frame = cv2.imread('Test Images/MultiColor.jpg')

cv2.imshow('mask',frame)

cv2.waitKey(0)

##INVERTS COLORS
frame = ~frame

# It converts the BGR color space of image to HSV color space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Threshold of blue in HSV space


sensitivity = 30
lower_inverted_white = np.array([90, 0, 100])
upper_inverted_white = np.array([130, 80, 255])


# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_inverted_white, upper_inverted_white)

##DETECTING THE HOLES IN EACH PIXEL
cv2.imshow('mask',mask)

cv2.waitKey(0)

contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = ~frame.copy()

for i in contours:
    x,y,w,h = cv2.boundingRect(i)

    if w*h < 1200 or w*h > 2500:
        continue
    print(w * h)
    cv2.rectangle(image_copy, (x-10, y-10), (x + w+10, y + h+10), (0,0,255), 4)
    colorsB = ~frame[y-7, x-7, 0]
    colorsG = ~frame[y-7, x-7, 1]
    colorsR = ~frame[y-7, x-7, 2]
    hsv_value = np.uint8([[[colorsB, colorsG, colorsR]]])
    hsv = cv2.cvtColor(hsv_value, cv2.COLOR_BGR2HSV)

    white_prob = ((lower_white <= hsv) & (hsv <= upper_white)).all()
    yellow_prob = ((lower_yellow <= hsv) & (hsv <= upper_yellow)).all()
    green_prob = ((lower_green <= hsv) & (hsv <= upper_green)).all()
    purple_prob = ((lower_purple <= hsv) & (hsv <= upper_purple)).all()

    putText(image_copy, "WHITE" if white_prob else ("YELLOW" if yellow_prob else ("GREEN" if green_prob else ("PURPLE" if purple_prob else "NONE"))), 255,0,0,x-10,y+20)


# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)


cv2.destroyAllWindows()