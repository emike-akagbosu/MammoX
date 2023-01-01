from django.db import models


class Mammogram(models.Model):
    '''Defines Mammogram class with Image Field'''
    #create image field and send image to upload folder
    image = models.ImageField(upload_to='api/static/Images/upload')
    