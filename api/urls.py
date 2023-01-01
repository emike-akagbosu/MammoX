
from django.urls import path
import api.views
from django.conf.urls.static import static
from django.conf import settings

# Add each page as URL 
urlpatterns = [
    path('', api.views.main),
    path('index.html', api.views.main),
    path('about.html', api.views.about),
    path('contact.html', api.views.contact),
    path('redirectedres.html', api.views.results),
 
]


urlpatterns += static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)