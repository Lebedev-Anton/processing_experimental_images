from django.urls import path
from .views import index, set_command

urlpatterns = [
    path('', index),
    path('set_command/', set_command)
]