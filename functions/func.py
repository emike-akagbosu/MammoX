import os
import joblib 
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from skimage.morphology import area_closing, area_opening
import shutil
import cv2
from skimage import io
from skimage.color import rgb2gray
import warnings
#warnings.filterwarnings("ignore")


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

directory = os.fsencode(os.path.join(ROOT_DIR, 'api/static/Images/upload'))


def final_classifier():
    for file in os.listdir(directory):
        filepath = os.path.realpath((os.path.join(directory, file))).decode("utf-8")
                                   
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

        return res2, RGB_values

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
    def get_properties(outputfinal):

        properties = ['area','convex_area',
                     'bbox_area','major_axis_length', 
                     'minor_axis_length', 'perimeter',  
                     'equivalent_diameter', 'mean_intensity',  
                     'solidity', 'eccentricity']
        dataframe = pd.DataFrame(columns=properties)
        try:
            grayscale = rgb2gray(outputfinal)
        except:
            grayscale = outputfinal
        threshold = threshold_otsu(grayscale)
        binarized = grayscale < threshold         
        closed = area_closing(binarized,1000)
        opened = area_opening(closed,1000)
        labeled = label(opened)
        regions = regionprops(labeled)
        data = pd.DataFrame(regionprops_table(grayscale, grayscale,
                            properties=properties))
        data = data[(data.index!=0) & (data.area>100)]
        dataframe = pd.concat([dataframe, data])
        return dataframe

    if (filepath).find("CC") != -1:
        outputfinal = remove_background(filepath)
    elif (filepath).find("MLO") != -1:
        output = remove_background(filepath)

        output_seg, RGB_values = process(output)
        threshold = RGB_values[2]+5
        outputfinal = remove_pect(output_seg, output, threshold)
        
    density = get_properties(outputfinal)
    density['type'] = 'unknown'
    #print("The shape of the dataframe is: ", density.shape)
    #display(density)

    dff = density
    dff['ratio_length'] = (dff['major_axis_length'] / 
                          dff['minor_axis_length'])
    dff['perimeter_ratio_major'] = (dff['perimeter'] /  
                                   dff['major_axis_length'])
    dff['perimeter_ratio_minor'] = (dff['perimeter'] /
                                   dff['minor_axis_length'])
    dff['area_ratio_convex'] = dff['area'] / dff['convex_area']
    dff['area_ratio_bbox'] = dff['area'] / dff['bbox_area']
    dff['peri_over_dia'] = dff['perimeter'] / dff['equivalent_diameter']
    final_dff = dff[dff.drop('type', axis=1).columns].astype(float)
    final_dff = final_dff.replace(np.inf, 0)
    X = final_dff
    
    #display(X)
    try:
        modeldir = os.fsencode(os.path.join(ROOT_DIR, 'RF_compressed.joblib'))
        modeldirMLO = os.fsencode(os.path.join(ROOT_DIR, 'RF_compressed_MLO.joblib'))

        if (filepath).find("CC") != -1:
                loaded_rf = joblib.load(modeldir)

        elif (filepath).find("MLO") != -1:
            loaded_rf = joblib.load(modeldirMLO)
        #print(loaded_rf.predict(X))
        final_output = loaded_rf.predict(X)
        sum = 0
        length = len(final_output)
        for x in final_output:
          sum = sum +int(x)
        #print(sum/length)
        final_predict = round(sum/length)
        #print(round(sum/length))#final prediction
        bands = [0,25,50,75,100]
        final_pct = '('+str(bands[final_predict-1]) + '% - ' +  str(bands[final_predict]) + '%)'
       
    except:
        final_predict = 'Error loading algorithm'
        final_pct = 'N/A'

    

    return(final_predict, final_pct)

def clear_img():
    if os.path.exists(directory) and os.path.isdir(directory):
        shutil.rmtree(directory)
    


# 1. Add input image to 'api/static/Images/upload', create upload folder if not there already
# 2. Uncomment this code to run the algorithm.
#final_result = final_classifier()
#print(final_result)