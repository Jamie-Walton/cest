# Generated by Django 4.2.3 on 2023-09-20 22:22

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("analyze", "0005_alter_file_scan"),
    ]

    operations = [
        migrations.RemoveField(model_name="file", name="scan",),
    ]