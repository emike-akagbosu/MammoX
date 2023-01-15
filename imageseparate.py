import numpy as np
import cv2
import os
import csv

import os
import pytz
import time

filepath = []
file = []

#function to remove background with otsu thresholding
def remove_background(filepath):
    image1 = cv2.imread(filepath)

    height = image1.shape[0]
    width = image1.shape[1]

    new_height = 720
    new_width = int(new_height / height * width)

    new_size = cv2.resize(image1, (new_width, new_height))

    hh, ww = new_size.shape[:2]

    img = cv2.cvtColor(new_size, cv2.COLOR_BGR2GRAY)

    # apply otsu thresholding
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    
    # Reference 1 - taken from https://stackoverflow.com/questions/67227335/how-to-remove-mammography-tag-artifacts
    # apply morphology close to remove small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # apply morphology open to separate breast from other regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # draw largest contour as white filled on black background as mask
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # apply mask to image
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # End of Reference 1
    
    return result


def process(input_image):
    # In opencv, images are read as BGR
    img = input_image
    # img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    height, width = img.shape

    # for k-means, we need to flatten the image
    # reshape image into different size
    img2 = img.reshape((-1, 1))

    img2 = np.float32(img2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Clusters
    k = 4

    attempts = 10

    # ret,label,center=cv.kmeans(img2,k,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)
    ret, label, centre = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    # Convert centres into unsigned integers
    centre = np.uint8(centre)

    res = centre[label.flatten()]
    res2 = res.reshape((img.shape))
    # cv2.imshow('Binary Threshold',res2)

    # ,centre[4][0],centre[5][0]
    RGB_values = [centre[0][0], centre[1][0], centre[2][0], centre[3][0]]
    RGB_values = np.sort(RGB_values)

    return res2, RGB_values

def remove_pect(input_image, original_bg, thresh_value):
    '''The input to this function has to be (segmented image, image with background removed, threshold)'''

    img = input_image
    img2 = original_bg

    hh, ww = img.shape[:2]

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)

    # apply morphology close to remove small regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # apply morphology open to separate breast from other regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # big_contour = min(contours, key=cv2.contourArea)
    big_contour = max(contours, key=cv2.contourArea)

    # draw largest contour as white filled on black background as mask
    mask = np.zeros((hh, ww), dtype=np.uint8)
    cv2.drawContours(mask, [big_contour], 0, 255, cv2.FILLED)

    # dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    inv_mask = np.invert(mask)

    # apply mask to image
    result = cv2.bitwise_and(img2, img2, mask=inv_mask)

    return result

    # density_percentage = (count3)/(img2.shape[0]-count1) * 100
    # print(density_percentage)


start = time.time()

#rootdir = 'C:/Users/emike/OneDrive - Imperial College London/Programming 3/case0202/'
rootdir = 'C:/temp_7zip/normals_png/'
#rootdir = 'C:/Users/Indum/Documents/Year3/Programming/project/csv_test/'
writedir = 'C:/temp_7zip'

#function goes through the root folder and checks each image, sees if its an MLO or CC image and then sorts it according to its density
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        strippath = subdir
        #finding ics file which contains information about the image such as density category
        for x in range(0, len(str(file))):
            if (file).find("ics") != -1:
                e = open(filepath, 'rb')
                read_e = str(e.read())
                findpos = read_e.find('DENSITY')
                densityno = int(read_e[findpos + 8])
                print(densityno)
                e.close()
                break
                 #loop stopped if file is found
                    
        #finding CC file        
        no = 1   
        for x in range(0, len(str(file))):
            if (file).find("CC") != -1:
                # Add lines to remove background first
                filename_without_ext = os.path.splitext(file)[0]
                output = remove_background(filepath)
                if no<2:
                  #print('writing')  
                #sorting images into correct folders
                  if densityno == 1:
                      cv2.imwrite(writedir+'/images/train/1CC/'+file,output)
                  elif densityno == 2:
                      cv2.imwrite(writedir+'/images/train/2CC/'+file,output)
                  elif densityno == 3:
                      cv2.imwrite(writedir+'/images/train/3CC/'+file ,output)
                  elif densityno == 4:
                      cv2.imwrite(writedir+'/images/train/4CC/'+file ,output)
                  no = no + 1
            if no == 2:
                break
                 #loop stopped if file is found
                    
        #finding MLO file       
        no = 1
        for x in range(0, len(str(file))):
            if (file).find("MLO") != -1:
                # Add lines to remove background first
                filename_without_ext = os.path.splitext(file)[0]
                output = remove_background(filepath)

                output_seg, RGB_values = process(output)
                #print(RGB_values[2],RGB_values[3])
                #threshold = int((RGB_values[2] + RGB_values[3]) / 2)
                #threshold = 130 --> this worked for both in case0202
                threshold = RGB_values[2]+5
                pect_removed = remove_pect(output_seg, output, threshold)
                #sorting images into correct folders
                if no<2:
                  #print('writing')  
                  if densityno == 1:
                      cv2.imwrite(writedir+'/images/train/1MLO/'+file,pect_removed)
                  elif densityno == 2:
                      cv2.imwrite(writedir+'/images/train/2MLO/'+file,pect_removed)
                  elif densityno == 3:
                      cv2.imwrite(writedir+'/images/train/3MLO/'+file ,pect_removed)
                  elif densityno == 4:
                      cv2.imwrite(writedir+'/images/train/4MLO/'+file ,pect_removed)
                  no = no + 1
            if no == 2:
                break
                #loop stopped if file is found


end = time.time()

print("The time of execution of above program is :",
      (end - start) * 10 ** 3, "ms")
