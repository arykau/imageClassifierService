from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('uploadImage', views.classify, name='uploadImage'),
]
