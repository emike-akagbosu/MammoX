import cv2
import numpy as np

path = 'C:\cygwin64\home\DDSM-LJPEG-Converter\cases2and3\case0002\A_0002_1.RIGHT_MLO_segmented.png'
image = cv2.imread(path)

#image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# apply otsu thresholding
#ret, thresh = cv2.threshold(img, 0, 255,cv2.THRESH_OTSU)

hh,ww = image.shape[:2]

img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
#ret, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)

# apply morphology close to remove small regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# apply morphology open to separate breast from other regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

# get largest contour
contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
#big_contour = min(contours, key=cv2.contourArea)
big_contour = max(contours, key=cv2.contourArea)

# draw largest contour as white filled on black background as mask
mask = np.zeros((hh,ww), dtype=np.uint8)
cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

# dilate mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
inv_mask = np.invert(mask)

# apply mask to image
result = cv2.bitwise_and(img, img, mask=inv_mask)

#cv2.imshow('thresh', thresh)
#cv2.imshow('morph', morph)
#cv2.imshow('mask', mask)
#cv2.imshow('Inverse mask', inv_mask)
cv2.imshow('result', result)

cv2.imwrite('C:\cygwin64\home\DDSM-LJPEG-Converter\cases2and3\case0002\A_0002_1.RIGHT_MLO_pect.png',result)
