from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from functions.func import final_classifier, clear_img
from .form import ImageForm
from .models import Mammogram


# Create your views here.
def main(request):
    ''' Home page view - creates form and runs ML model on input image'''
    clear_img()
    if request.method == 'POST': 
        form=ImageForm(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.save()
            obj=form.instance ## Form code from "add ref"
            band = final_classifier()
            
            return render(request,"results.html", {"display_band":band, "img":obj})
    else:
        form=ImageForm()
 
    
    return render(request,"index.html",{"form":form})

def results(request):
    img=Mammogram.objects.last()
 
    return render(request,"results.html",{"img":img})

def about(request):
    return render(request,"about.html")

def contact(request):
    return render(request,"contact.html")



    



