# Generated by Django 4.0.2 on 2022-02-28 19:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_setting', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='setting',
            name='white_balance',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
