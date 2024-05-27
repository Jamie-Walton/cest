from django.db import models

def directory_path(instance, filename):
    return 'uploads/{0}/{1}'.format(instance.dataset.identifier, filename)

def image_path(instance, filename):
    return 'uploads/{0}/images/{1}'.format(instance.dataset.identifier, filename)

class Dataset(models.Model):

    identifier = models.CharField(max_length=20)
    owner = models.CharField(max_length=50, blank=True)
    institution = models.CharField(max_length=50, blank=True)
    image_width = models.FloatField(default=0)
    image_height = models.FloatField(default=0)
    
    def __str__(self):
        return f'Dataset<{self.identifier}>'


class File(models.Model):

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    file = models.FileField(upload_to=directory_path, blank=True, null=True)
