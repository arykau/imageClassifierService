from distutils.command.upload import upload
from email.mime import image
from django.db import models

# Create your models here.


class Result(models.Model):
    img_path = models.TextField()
    classification = models.TextField()

    def __str__(self):
        return self.img_path + " " + self.image
