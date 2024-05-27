# Generated by Django 4.2.3 on 2023-09-20 20:42

import analyze.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("analyze", "0003_auto_20230405_0203"),
    ]

    operations = [
        migrations.AddField(
            model_name="file",
            name="scan",
            field=models.ImageField(blank=True, null=True, upload_to="None"),
        ),
        migrations.AlterField(
            model_name="file",
            name="file",
            field=models.FileField(
                blank=True, null=True, upload_to=analyze.models.directory_path
            ),
        ),
    ]