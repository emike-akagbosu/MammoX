##importing the Image class from PIL package
#from PIL import Image
##read the image, creating an object
#im = Image.open(r"C:\Users\sonya\Documents\y3_prog\MammoX\mammoxdev-env\mammox\files\input\breast-calcification-mammogram.jpg")
##show picture
#im.show()


##import the cv2 module.
#import cv2 as cv
##imread method loads the image. We can use a relative path if 
##picture and python file are in the same folder
#img = cv.imread('C:\Users\sonya\Documents\y3_prog\MammoX\mammoxdev-env\mammox\files\input\breast-calcification-mammogram.jpg')
##method resize is used to modify de size of the picture
#imS = cv.resize(img, (960, 540))
##We use imshow method to show the picture
#cv.imshow('Picture of trees',imS)
##If we don’t use the waitKey method, picture
##show and disappears immediately.
#cv.waitKey(0)
##destroyallwindows used to free up resources
#cv.destroyAllWindows()