from django.db import models


class Command(models.Model):
    command = models.CharField(max_length=50)
    parameter = models.JSONField()
    status = models.BooleanField()
