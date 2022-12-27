import numpy as np
import cv2
import os
import csv

import os
import pytz
import time

filepath = []
file = []

def remove_background(filepath):
    image1 = cv2.imread(filepath)

    height = image1.shape[0]
    width = image1.shape[1]

    new_height = 720
    new_width = int(new_height / height * width)

    new_size = cv2.resize(image1,(new_width,new_height))

    hh,ww = new_size.shape[:2]

    img = cv2.cvtColor(new_size, cv2.COLOR_BGR2GRAY)

    # apply otsu thresholding
    ret, thresh = cv2.threshold(img, 0, 255,cv2.THRESH_OTSU)

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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100,100))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)

    return result

start = time.time()   

#rootdir = 'C:/Users/emike/OneDrive - Imperial College London/Programming 3/case0202/'
#rootdir = 'C:/cygwin64/home/DDSM-LJPEG-Converter/cases2and3/'
#rootdir = 'C:/Users/Indum/Documents/Year3/Programming/project/csv_test/'
rootdir = 'C:/temp_7zip/normals_png'
writedir = 'C:/temp_7zip'

number = 1390
percentagetrain = 0.6
traindatano = number*percentagetrain
traincount = 0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        strippath = subdir
        #print(file)
        no = 1
        for x in range(0,len(str(file))):
            if (file).find("ics")!=-1:
                e = open(filepath, 'rb')
                read_e = str(e.read())
                findpos = read_e.find('DENSITY')
                densityno = int(read_e[findpos+8])
                e.close()
                #print('found ics',densityno)
                break
        for x in range(0,len(str(file))):
            if (file).find("CC")!=-1:
                  output = remove_background(filepath)
                 #print('found cc')
                  if no<3:
                      #if traincount< traindatano:
                      if densityno == 1:
                          cv2.imwrite(writedir+'/images/train/1/'+file,output)
                      elif densityno == 2:
                          cv2.imwrite(writedir+'/images/train/2/'+file,output)
                      elif densityno == 3:
                          cv2.imwrite(writedir+'/images/train/3/'+file ,output)
                      elif densityno == 4:
                          cv2.imwrite(writedir+'/images/train/4/'+file ,output)
##                      elif:
##                          print('all train data done, now test data')
##                          if densityno == 1:
##                              cv2.imwrite(rootdir+'/images/test/1/'+file,output)
##                          elif densityno == 2:
##                              cv2.imwrite(rootdir+'/images/test/2/'+file,output)
##                          elif densityno == 3:
##                              cv2.imwrite(rootdir+'/images/test/3/'+file ,output)
##                          elif densityno == 4:
##                              cv2.imwrite(rootdir+'/images/test/4/'+file ,output)                                                      
                  if no == 3:
                      break
                  no= no + 1
    print(traincount)              
    traincount = traincount + 1
                  

                   

end = time.time()

print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")
