# Generated by Django 3.2.5 on 2023-04-04 23:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analyze', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataset',
            name='files',
            field=models.FileField(blank=True, null=True, upload_to='uploads/<function directory_path at 0x7fb898d23c10>/'),
        ),
    ]