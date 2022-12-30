import joblib
import numpy as np
import pandas as pd
import cv2
import os
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from skimage.morphology import area_closing, area_opening
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#directory for user uploaded image
rootdir = 'C:/Users/Indum/Documents/Year3/Programming/project/final/test_image/D_4595_1.LEFT_CC.png'
#directory for model file
modeldir = "C:/Users/Indum/Documents/Year3/Programming/project/final/random_forest.joblib"

def final_classifer(rootdir,modeldir):
    #this function gets the properties of the uploaded image
    def get_properties(rootdir):
        file = rootdir
        #properties
        properties = ['area','convex_area',
                     'bbox_area','major_axis_length', 
                     'minor_axis_length', 'perimeter',  
                     'equivalent_diameter', 'mean_intensity',  
                     'solidity', 'eccentricity']
        #arranges data in a table using pandas
        dataframe = pd.DataFrame(columns=properties)
        try:
          #error handling to check uploaded file is the correct format
          grayscale = rgb2gray(imread(file))
        except:
           #if not correct format user is notified
          print("Wrong input format")
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
        #dataset with all the values returned
        return dataframe

    density = get_properties(rootdir)
    density['type'] = 'unknown'
    print("The shape of the dataframe is: ", density.shape)
    #display(density)

    #further properties calculated and added for higher model accuracy
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

    #model is loaded
    loaded_rf = joblib.load(modeldir)
    #print(loaded_rf.predict(X))
    
    #model used to predict density for each region in the image
    final_output = loaded_rf.predict(X)
    sum = 0
    length = len(final_output)
    for x in final_output:
      sum = sum +int(x)
    #print(sum/length)
    
    #average is calculated for each region in the image and rounded for prediction of the density
    final_predict = round(sum/length)
    #print(round(sum/length))#final prediction
    return(final_predict)

#main function that is called
final_predict = final_classifer(rootdir,modeldir)
