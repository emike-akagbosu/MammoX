from django.db import models

# Create your models here.
class Mammogram(models.Model):
    image = models.ImageField(upload_to='files/input')
    