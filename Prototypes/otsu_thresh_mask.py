import cv2
import numpy as np
import matplotlib.pyplot as plt

# Code for resizing image and removing tag artefacts

path = 'C:\cygwin64\home\DDSM-LJPEG-Converter\case0202\A_0202_1.LEFT_CC.png'
#path = 'C:\cygwin64\home\DDSM-LJPEG-Converter\case0202\A_0202_1.LEFT_MLO.png'
#path = 'C:\cygwin64\home\DDSM-LJPEG-Converter\case0202\A_0202_1.RIGHT_CC.png'
#path = 'C:\cygwin64\home\DDSM-LJPEG-Converter\case0202\A_0202_1.RIGHT_MLO.png'
image1 = cv2.imread(path)

height = image1.shape[0]
width = image1.shape[1]

new_height = 720
new_width = int(new_height / height * width)

new_size = cv2.resize(image1,(new_width,new_height))

hh,ww = new_size.shape[:2]

img = cv2.cvtColor(new_size, cv2.COLOR_BGR2GRAY)

# apply otsu thresholding
ret, thresh = cv2.threshold(img, 0, 255,cv2.THRESH_OTSU)

###Start of test region

# apply morphology close to remove small regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# apply morphology open to separate breast from other regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# get largest contour
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)

# draw largest contour as white filled on black background as mask
mask = np.zeros((hh,ww), dtype=np.uint8)
cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

# dilate mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

# apply mask to image
result = cv2.bitwise_and(img, img, mask=mask)

###End of test region
#winname = 'Binary Threshold'
#cv2.namedWindow(winname)
#cv2.moveWindow(winname,40,30)
cv2.imshow('thresh', thresh)
cv2.imshow('morph', morph)
cv2.imshow('mask', mask)
cv2.imshow('result', result)

cv2.imwrite('C:\cygwin64\home\DDSM-LJPEG-Converter\case0202\A_0202_1.LEFT_CC_thresh.png',thresh)
cv2.imwrite('C:\cygwin64\home\DDSM-LJPEG-Converter\case0202\A_0202_1.LEFT_CC_mask.png',mask)
cv2.imwrite('C:\cygwin64\home\DDSM-LJPEG-Converter\case0202\A_0202_1.LEFT_CC_result.png',result)

# Hold window on screen until close input received
cv2.waitKey(0)
# Delete created GUI window from screen and memory
cv2.destroyAllWindows()
