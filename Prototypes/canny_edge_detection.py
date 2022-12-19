import cv2
import numpy as np
from matplotlib import pyplot as plt

path = 'C:\cygwin64\home\DDSM-LJPEG-Converter\cases2and3\case0002\A_0002_1.RIGHT_MLO_result2.png'
img = cv2.imread(path,0)
edges = cv2.Canny(img,100,110)

plt.subplot(2,1,1),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(edges,cmap = 'gray')
plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

plt.show()
