import numpy as np
import cv2
import os
import csv

import icalendar
import os
import pytz

filepath = []
file = []

def process(filepath,file):
    # In opencv, images are read as BGR
    img = cv2.imread(filepath)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, other = img.shape

    # for k-means, we need to flatten the image
    # reshape image into different size
    img2 = img.reshape((-1,3))
    #print(img2.shape)

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

    # Lower RGB number means darker
    array_to_add = [file,count1,count2,count3,count4,count5,count6]

    return(array_to_add)                

    # This can be used in the grayscale case
    # np.count_nonzero(img == value)

    #density_percentage = (count3)/(img2.shape[0]-count1) * 100
    #print(density_percentage)


#def extract_density(filepath,file)    

rootdir = 'C:/Users/Indum/Documents/Year3/Programming/project/csv_test/'

f = open(rootdir + 'csv_file', 'w')
header = ['picture', 'count1', 'count2', 'count3', 'count4', 'count5', 'count6','density']
writer = csv.writer(f)
writer.writerow(header)
f.close()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        for x in range(0,len(str(file))):
            if (file).find("ics")!=-1:
                e = open(filepath, 'rb')
                read_e = str(e.read())
                findpos = read_e.find('DENSITY')
                densityno = read_e[findpos+8]
                e.close()
                break
                
            if (file).find("CC")!=-1:
               result_array = process(filepath,file)
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
         
         
