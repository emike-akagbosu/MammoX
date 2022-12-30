from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from functions.func import final_classifier
from .form import ImageForm
from .models import Mammogram


# Create your views here.
def main(request):
    if request.method == 'POST': 
        form=ImageForm(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.save()
            obj=form.instance
            display = final_classifier()
  
            
        return render(request,"results.html", {"display":display, "img":obj})

            #render(request,"index.html",{"obj":obj})
    else:
        form=ImageForm()
 
    
    return render(request,"index.html",{"form":form})

def results(request):
    
    img=Mammogram.objects.last()
 
    return render(request,"results.html",{"img":img})

def redirect(request):

    return render(request,"redirect.html")

def about(request):
    return render(request,"about.html")

def contact(request):
    return render(request,"contact.html")



    



