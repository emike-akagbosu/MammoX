# Generated by Django 4.1.4 on 2022-12-29 18:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mammogram',
            name='image',
            field=models.ImageField(upload_to='api/static/Images'),
        ),
    ]