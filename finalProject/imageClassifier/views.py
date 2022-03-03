from fileinput import filename
import re
from tkinter import image_names
from urllib import response
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from imageClassifier.models import Result

import numpy as np
import os
import cgi
import PIL
import tensorflow as tf

from keras import layers, models, datasets
from keras.models import Sequential
from keras.preprocessing import image

from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

# Create your views here.


def index(request):
    # cifar()

    return render(request, 'index.html')


def classify(request):

    classified = get_path(request)

    return render(request, 'index.html', classified)


def get_path(request):

    if request.method == 'POST':

        image = request.FILES['image_input']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        filepath = fs.url(filename)

        classified = get_model(image.name)

        context = {
            "class": classified,
            "filename": filepath,
        }

        res = Result(img_path=filepath, classification=classified)
        res.save()

        return context


def get_model(filename):

    model = tf.keras.models.load_model("myModels/model.h5")

    img_url = 'media/' + filename

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    img = tf.keras.utils.load_img(
        img_url, target_size=(32, 32))

    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

    return class_names[np.argmax(score)]


def cifar(self):
    (train_images, train_labels), (test_images,
                                   test_labels) = cifar10.load_data()

    train_images, test_images = train_images / 255.0, test_images / 255.0

    model = models.Sequential()
    model.add(layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())

    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels))
    model.save("my_model")
    model.save("myModels/model.h5")
    return model
