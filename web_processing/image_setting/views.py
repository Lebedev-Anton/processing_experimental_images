from django.shortcuts import render, redirect, reverse
from .models import Setting
from api.models import Command
from django.http import FileResponse
import os


def index(request):
    setting = Setting.objects.order_by('-pk').first()
    return render(request, 'index.html', {'setting': setting})


def make_photo(request):
    ISO = request.GET.get('ISO')
    wb = request.GET.get('wb')

    setting = Setting(ISO=ISO, white_balance=wb)
    setting.save()

    command = Command(command='set_ISO', parameter=ISO, status=False)
    command.save()

    command = Command(command='set_white_balance', parameter=wb, status=False)
    command.save()

    command = Command(command='capture_image', parameter='None', status=False)
    command.save()
    return redirect('index')


def download(request):
    print('ewtyrhy')
    path = os.path.abspath(os.path.join('..', 'web_processing', 'image_setting', 'static', 'TEST2.JPG'))
    response = FileResponse(open(path, 'rb'))
    return response

