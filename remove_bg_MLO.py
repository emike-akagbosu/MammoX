import numpy as np
import cv2
import os
import csv

# import icalendar
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

    new_size = cv2.resize(image1, (new_width, new_height))

    hh, ww = new_size.shape[:2]

    img = cv2.cvtColor(new_size, cv2.COLOR_BGR2GRAY)

    # apply otsu thresholding
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

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

    # Count the number of pixels in each cluster
    count1 = np.count_nonzero(res2 == RGB_values[0])  # Darkest
    count2 = np.count_nonzero(res2 == RGB_values[1])
    count3 = np.count_nonzero(res2 == RGB_values[2])
    count4 = np.count_nonzero(res2 == RGB_values[3])
    # count5 = np.count_nonzero(res2 == RGB_values[4])
    # count6 = np.count_nonzero(res2 == RGB_values[5])

    totalno = count2 + count3 + count4  # +count5+count6

    # Lower RGB number means darker
    array_to_add = [file, count1, count2, count3, count4]  # , count5, count6]
    for x in range(1, len(array_to_add)):
        array_to_add[x] = (array_to_add[x] / (totalno)) * 100
    # density_percentage = (count6 + count5 + count4) / (totalno) * 100  # percentage nonfat to fat, using lower threshold

    return array_to_add, res2, RGB_values

    # density_percentage = (count3)/(img2.shape[0]-count1) * 100
    # print(density_percentage)


def remove_pect(input_image, original_bg, thresh_value):
    '''The input to this function has to be (segmented image, image with background removed)'''

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
rootdir = 'C:/cygwin64/home/DDSM-LJPEG-Converter/cases/'
# rootdir = 'C:/Users/Indum/Documents/Year3/Programming/project/csv_test/'

f = open(rootdir + 'csv_file', 'w')
header = ['picture', 'count1', 'count2', 'count3', 'count4', 'count5', 'count6', 'density']
writer = csv.writer(f)
writer.writerow(header)
f.close()

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = os.path.join(subdir, file)
        strippath = subdir
        for x in range(0, len(str(file))):
            if (file).find("ics") != -1:
                e = open(filepath, 'rb')
                read_e = str(e.read())
                findpos = read_e.find('DENSITY')
                densityno = read_e[findpos + 8]
                e.close()
                break

            if (file).find("CC") != -1:
                # Add lines to remove background first
                filename_without_ext = os.path.splitext(file)[0]
                output = remove_background(filepath)

                # Write file for CC image with bg removed
                cv2.imwrite(strippath + '/' + filename_without_ext + '_bg.png', output)
                result_array, output_seg, RGB_values = process(output)
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

            if (file).find("MLO") != -1:
                # Add lines to remove background first
                filename_without_ext = os.path.splitext(file)[0]
                output = remove_background(filepath)

                result_array, output_seg, RGB_values = process(output)
                print(RGB_values[2],RGB_values[3])
                #threshold = int((RGB_values[2] + RGB_values[3]) / 2)
                #threshold = 130 --> this worked for both in case0202
                threshold = RGB_values[2]+5
                pect_removed = remove_pect(output_seg, output, threshold)

                result_array, output_seg2, values = process(pect_removed)

                # Write file for MLO image with bg removed and pect tissue removed
                cv2.imwrite(strippath + '/' + filename_without_ext + '_bg.png', pect_removed)

                #Create result array
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


end = time.time()

print("The time of execution of above program is :",
      (end - start) * 10 ** 3, "ms")