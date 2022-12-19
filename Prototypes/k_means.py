import numpy as np
import cv2

# See documentation for more on k-means algorithm in opencv

# In opencv, images are read as BGR
path = 'C:\cygwin64\home\DDSM-LJPEG-Converter\cases2and3\case0002\A_0002_1.LEFT_CC_result2.png'
img = cv2.imread(path)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width, other = img.shape

# for k-means, we need to flatten the image
# reshape image into different size
img2 = img.reshape((-1,3))
print(img2.shape)

img2 = np.float32(img2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Clusters
k = 4

attempts = 10

#ret,label,center=cv.kmeans(img2,k,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)
ret,label,centre=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

# Convert centres into unsigned integers
centre = np.uint8(centre)

res = centre[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imwrite('C:\cygwin64\home\DDSM-LJPEG-Converter\cases2and3\case0002\A_0002_1.LEFT_CC_segmented_k4.png',res2)

RGB_values= [centre[0][0], centre[1][0], centre[2][0], centre[3][0]]
RGB_values = np.sort(RGB_values)

# Count the number of pixels in each cluster
count1 = np.count_nonzero((res2 == RGB_values[0]).all(axis = 2)) # Darkest
count2 = np.count_nonzero((res2 == RGB_values[1]).all(axis = 2))
count3 = np.count_nonzero((res2 == RGB_values[2]).all(axis = 2))
count4 = np.count_nonzero((res2 == RGB_values[3]).all(axis = 2))

# Lower RGB number means darker

print(count1,count2,count3,count4)

# This can be used in the grayscale case
# np.count_nonzero(img == value)

#density_percentage = (count3)/(img2.shape[0]-count1) * 100
#print(density_percentage)
