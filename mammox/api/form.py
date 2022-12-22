from django import forms
from .models import Mammogram

class ImageForm(forms.ModelForm):
    class Meta:
        model=Mammogram
        fields=('image',)