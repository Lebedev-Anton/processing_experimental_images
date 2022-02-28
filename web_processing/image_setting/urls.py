from django.urls import path
from .views import index, make_photo, download

urlpatterns = [
    path('', index, name='index'),
    path('make_photo/', make_photo, name='make_photo'),
    path('download/', download, name='download'),
]