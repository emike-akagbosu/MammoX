import os
import unittest



class TestStringMethods(unittest.TestCase):

    def test_checkInputFormat(self):  # check that image is png format
        rootdir = 'C:/Users/sarah/OneDrive/Dokumente/case0002'
        with os.scandir(rootdir) as files:
            for file in files:
                if("CC" in file.name or "MLO" in file.name):
                    self.assertTrue(file.name.endswith("png")) 
                        
    def test_checkCCFilesTrain(self): # check that only CC images in CC Train folders
        rootdir = 'C:/Users/sarah/OneDrive/Dokumente/processed_im/images/train'
        with os.scandir(rootdir) as subdirectories:
            for dir in subdirectories:
                if("CC" in dir.name):
                    with os.scandir(rootdir + '/' + dir.name) as entries:
                        for file in entries:
                            self.assertTrue("CC" in file.name) 

    
    def test_checkMLOFilesTrain(self):  # check that only MLO images in MLO Train folders
        rootdir = 'C:/Users/sarah/OneDrive/Dokumente/processed_im/images/train'
        with os.scandir(rootdir) as subdirectories:
            for dir in subdirectories:
                if("MLO" in dir.name):
                    with os.scandir(rootdir + '/' + dir.name) as entries:
                        for file in entries:
                            self.assertTrue("MLO" in file.name) 
        

if __name__ == '__main__':
    unittest.main()
