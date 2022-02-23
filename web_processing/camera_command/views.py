from django.http import JsonResponse
from .models import Command
from .serializers import CommandSerializer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt


@api_view(['GET'])
def index(request):
    if request.method == 'GET':
        commands = Command.objects.all()
        serializer = CommandSerializer(commands, many=True)
        return Response(serializer.data)


@api_view(['POST'])
def set_command(request):
    if request.method == 'POST':
        serializer = CommandSerializer(data=request.data)
        print("++++++++++++++++++++++++++++++++++++++++")
        print(request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)