from django.urls import path
from .views import command

urlpatterns = [
    path('command/', command)
]