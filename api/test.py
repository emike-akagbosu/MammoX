from django.test import TestCase
import os
import unittest
from django.test import Client
from functions.func import final_classifier, clear_img
import io
from  api.views import main
from PIL import Image

from django.urls import reverse

from rest_framework import status



ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
directory = os.fsencode(os.path.join(ROOT_DIR, 'api/static/Images/upload'))

# Create your tests here.
class ProjectTests(TestCase):
    def setUp(self):
        # Every test needs a client.
        self.client = Client()

    def test_homepage(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
                        
    def test_about(self):
        # Issue a GET request.
        response = self.client.get('/about.html')

        # Check that the response is 200 OK.
        self.assertEqual(response.status_code, 200)

    def test_contact(self):
        # Issue a GET request.
        response = self.client.get('/contact.html')

        # Check that the response is 200 OK.
        self.assertEqual(response.status_code, 200) 

    
    def test_ml_output(self):
        output = final_classifier()
        self.assertEquals(type(output), 'integer')   
        
    def test_delete_img(self):
        clear_img()
        self.assertEqual(os.path.exists(directory) and os.path.isdir(directory), 0)
    

    ##https://gist.github.com/guillaumepiot/817a70706587da3bd862835c59ef584e
    def generate_photo_file(self):
        file = io.BytesIO()
        image = Image.new('RGBA', size=(100, 100), color=(155, 0, 0))
        image.save(file, 'png')
        file.name = 'test.png'
        file.seek(0)
        return file

    def test_upload_photo(self):
        """
        Test if we can upload a photo
        """

        url = reverse(main)

        photo_file = self.generate_photo_file()

        data = {
                'photo':photo_file
            }

        response = self.client.post(url, data, format='multipart')
        self.assertEqual(response.status_code, status.HTTP_200_OK)           

if __name__ == '__main__':
    unittest.main()












#test upload

#test redirects

