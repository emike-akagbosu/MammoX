import numpy as np
import cv2
import os
import csv
import tqdm

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
    k = 8
    attempts = 10

    #ret,label,center=cv.kmeans(img2,k,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)
    ret,label,centre=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    # Convert centres into unsigned integers
    centre = np.uint8(centre)

    res = centre[label.flatten()]
    res2 = res.reshape((img.shape))
    res3 = cv2.resize(res2, (400, 600))
    cv2.imshow('K = 8',res3)
    cv2.waitKey(0)

    RGB_values= [centre[0][0], centre[1][0], centre[2][0], centre[3][0],centre[4][0],centre[5][0], centre[6][0],centre[7][0]]
    RGB_values = np.sort(RGB_values)

    # Count the number of pixels in each cluster
    count1 = np.count_nonzero((res2 == RGB_values[0]).all(axis = 2)) # Darkest
    count2 = np.count_nonzero((res2 == RGB_values[1]).all(axis = 2))
    count3 = np.count_nonzero((res2 == RGB_values[2]).all(axis = 2))
    count4 = np.count_nonzero((res2 == RGB_values[3]).all(axis = 2))
    count5 = np.count_nonzero((res2 == RGB_values[4]).all(axis = 2)) 
    count6 = np.count_nonzero((res2 == RGB_values[5]).all(axis = 2))
    count7 = np.count_nonzero((res2 == RGB_values[6]).all(axis = 2))
    count8 = np.count_nonzero((res2 == RGB_values[7]).all(axis = 2))


    totalno = count1+ count2 +count3 +count4+count5+count6+count7+count8

    # Lower RGB number means darker
    array_to_add = [file,count1,count2,count3,count4,count5,count6, count7, count8]
    for x in range(1,len(array_to_add)):
                   array_to_add[x] = (array_to_add[x]/totalno)*100

    density_percentage = (count7+count8)/(totalno-count1) * 100 #percentage nonfat to fat, using lower threshold
    dens_perc = (count8)/(totalno-count1) * 100 #percentage nonfat to fat, using higher threshold
    print(density_percentage)
    print(dens_perc)
    return(array_to_add)               

    # This can be used in the grayscale case
    # np.count_nonzero(img == value)

    


#def extract_density(filepath,file)    

rootdir = 'C:/Users/sarah/OneDrive/Dokumente/MammoX/test'

f = open(rootdir + 'csv_file', 'w')
header = ['picture', 'count1', 'count2', 'count3', 'count4', 'count5', 'count6', 'count7','count8','density']
writer = csv.writer(f)
writer.writerow(header)
f.close()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        print(filepath)
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
         
         
