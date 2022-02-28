from django.db import models


class Setting(models.Model):
    ISO = models.CharField(max_length=50)
    white_balance = models.CharField(max_length=50, blank=True, null=True)

