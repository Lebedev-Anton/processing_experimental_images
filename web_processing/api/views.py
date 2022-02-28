from .models import Command
from .serializers import CommandSerializer
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view
import os


@api_view(['GET', 'POST', 'DELETE'])
def command(request):
    if request.method == 'GET':
        commands = Command.objects.all()
        serializer = CommandSerializer(commands, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        f = request.FILES['files']
        path = os.path.abspath(os.path.join('..', 'web_processing', 'image_setting', 'static', 'TEST2.JPG'))
        with open(path, 'wb+') as destination:
            for chunk in f.chunks():
                destination.write(chunk)
        return Response(status=status.HTTP_200_OK)
    elif request.method == 'DELETE':
        print(request.data)
        commands = Command.objects.get(pk=request.data['id'])
        print(commands)
        commands.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
