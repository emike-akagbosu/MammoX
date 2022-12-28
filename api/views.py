from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from functions.func import calculate
from .form import ImageForm
from .models import Mammogram


# Create your views here.
def main(request):
    if request.method == 'POST': 
        form=ImageForm(data=request.POST,files=request.FILES)
        if form.is_valid():
            form.save()
            obj=form.instance
            return redirect("results.html")
            #render(request,"index.html",{"obj":obj})
    else:
        form=ImageForm()
    img=Mammogram.objects.all()
    
    return render(request,"index.html",{"img":img,"form":form})

def results(request):
    display = calculate(3,2)

    return render(request,"results.html", {"display":display})

def about(request):
    return render(request,"about.html")

def contact(request):
    return render(request,"contact.html")



    



