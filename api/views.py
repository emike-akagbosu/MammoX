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
            obj=form.instance ## Lines 12-16 "https://drive.google.com/file/d/1tp89cLBTYsFHuJtZgcjnt1dAQ2jv7q9l/view"
            band, pct = final_classifier()
            
            return render(request,"redirectedres.html", {"display_band":band, "display_pct":pct, "img":obj})
    else:
        form=ImageForm()
 
    
    return render(request,"index.html",{"form":form})

def results(request):
    img=Mammogram.objects.last()
 
    return render(request,"redirectedres.html",{"img":img})

def about(request):
    return render(request,"about.html")

def contact(request):
    return render(request,"contact.html")



    



