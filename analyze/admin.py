from django.contrib import admin
from analyze.models import Dataset

class DatasetAdmin(admin.ModelAdmin):
    pass

admin.site.register(Dataset, DatasetAdmin)
