import numpy as np
import cv2
import os
import csv

#import icalendar
import os
import pytz

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

    #cv2.imwrite('C:\cygwin64\home\DDSM-LJPEG-Converter\cases2and3\case0003\A_0003_1.RIGHT_MLO_result2.png',result)

    #new_file_name = filepath + file + '_bg.png'
    #cv2.imwrite(new_file_name,result)

    return result

def process(filepath,file):
    # In opencv, images are read as BGR
    img = cv2.imread(filepath)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, other = img.shape

    # for k-means, we need to flatten the image
    # reshape image into different size
    img2 = img.reshape((-1,3))

    img2 = np.float32(img2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Clusters
    k = 6

    attempts = 10

    #ret,label,center=cv.kmeans(img2,k,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)
    ret,label,centre=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    # Convert centres into unsigned integers
    centre = np.uint8(centre)

    res = centre[label.flatten()]
    res2 = res.reshape((img.shape))
    #cv2.imshow('Binary Threshold',res2)

    RGB_values= [centre[0][0], centre[1][0], centre[2][0], centre[3][0],centre[4][0],centre[5][0]]
    RGB_values = np.sort(RGB_values)

    # Count the number of pixels in each cluster
    count1 = np.count_nonzero((res2 == RGB_values[0]).all(axis = 2)) # Darkest
    count2 = np.count_nonzero((res2 == RGB_values[1]).all(axis = 2))
    count3 = np.count_nonzero((res2 == RGB_values[2]).all(axis = 2))
    count4 = np.count_nonzero((res2 == RGB_values[3]).all(axis = 2))
    count5 = np.count_nonzero((res2 == RGB_values[4]).all(axis = 2)) 
    count6 = np.count_nonzero((res2 == RGB_values[5]).all(axis = 2))

    totalno = count1+ count2 +count3 +count4+count5+count6

    # Lower RGB number means darker
    array_to_add = [file,count1,count2,count3,count4,count5,count6]
    for x in range(1,len(array_to_add)):
                   array_to_add[x] = (array_to_add[x]/totalno)*100

    return array_to_add, res2             


    #density_percentage = (count3)/(img2.shape[0]-count1) * 100
    #print(density_percentage)


#def extract_density(filepath,file)    

rootdir = 'C:/Users/emike/OneDrive - Imperial College London/Programming 3/case0202/'
#rootdir = 'C:/cygwin64/home/DDSM-LJPEG-Converter/cases2and3/'
#rootdir = 'C:/Users/Indum/Documents/Year3/Programming/project/csv_test/'

f = open(rootdir + 'csv_file', 'w')
header = ['picture', 'count1', 'count2', 'count3', 'count4', 'count5', 'count6','density']
writer = csv.writer(f)
writer.writerow(header)
f.close()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        strippath = subdir
        for x in range(0,len(str(file))):
            if (file).find("ics")!=-1:
                e = open(filepath, 'rb')
                read_e = str(e.read())
                findpos = read_e.find('DENSITY')
                densityno = read_e[findpos+8]
                e.close()
                break
                
            if (file).find("CC")!=-1:
               #Add lines to remove background first
               filename_without_ext = os.path.splitext(file)[0]
               output = remove_background(filepath)
               new_filepath = strippath + '/' + filename_without_ext + '_bg.png'
               cv2.imwrite(new_filepath,output)
               new_file = filename_without_ext + '_bg.png'

               result_array,output_seg = process(new_filepath,new_file)
               seg_filepath = strippath + '/' + filename_without_ext + '_seg.png'
               cv2.imwrite(seg_filepath,output_seg)
               
               result_array.append(densityno)
               print(result_array)
               f = open(rootdir + 'csv_file', 'a')
               # create the csv writer
               writer = csv.writer(f)
               # write a row to the csv file
               writer.writerow(result_array)
               # close the file
               f.close()
               break
