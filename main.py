from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import numpy as np
from werkzeug import secure_filename
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
import operator
# import imageio

# for regular expressions, saves time dealing with string data
import re
# system level operations (like loading files)
import sys
# for reading operating system data
import os

# tell our app where our saved model is


# initalize our flask app
app = Flask(__name__)
# global vars for easy reuseability
global model, graph

class_names = ['2084', '40219', '40220', '40223', '40231', '40236', '40243', '40244', '40254',
               '40261', '40282', '40283', '40300', '40626', '50072', '50092', '50096',
               '80040', '80106', '80108', '8108', '8602', '8765', '8834_8835', 'ART 40114', 'BGR3534']

# Make transforms and use data loaders

# We'll use these a lot, so make them variables
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

chosen_transforms = {'train': transforms.Compose([
    transforms.RandomResizedCrop(size=256),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
]), 'val': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean_nums, std_nums)
]),
}

model = torch.load("modelv1.pt")
model.eval()


def predict_image(image_path):
    print("Prediction in progress")
    image = Image.open(image_path)

    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_nums, std_nums)

    ])

    # Preprocessed the image
    image_tensor = transformation(image).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    # Turn the input into a Variable
    input = Variable(image_tensor)

    # Predict the class of the image
    output = model(input)
    # print(output)
    sm = torch.nn.Softmax()
    probabilities = sm(output)
    # print(probabilities)
    index = output.data.numpy().argmax()

    return index, probabilities


photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)


@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        os.rename('./' + filename, './' + 'output.png')

    print("debug")
    # read the image into memory
    s1, probabilities = predict_image('./output.png')
    prob_dict = {}
    probabilities = probabilities.tolist()
    # print(type(probabilities))
    print(len(probabilities))
    # print(probabilities)
    for i in range(len(class_names)):
        prob_dict[class_names[i]] = probabilities[0][i]

    sorted_dict = sorted(prob_dict.items(), key=operator.itemgetter(1), reverse=True)
    s1 = sorted_dict[0][0]
    s2 = round(((sorted_dict[0][1]) * 100), 2)
    s3 = sorted_dict[1][0]
    s4 = round(((sorted_dict[1][1]) * 100), 2)
    s5 = sorted_dict[2][0]
    s6 = round(((sorted_dict[2][1]) * 100), 2)
    s7 = sorted_dict[3][0]
    s8 = round(((sorted_dict[3][1]) * 100), 2)
    s9 = sorted_dict[4][0]
    s10 = round(((sorted_dict[4][1]) * 100), 2)

    # compute a bit-wise inversion so black becomes white and vice versa

    # convert the response to a string
    return render_template("index2.html", s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6, s7=s7, s8=s8, s9=s9, s10=s10)



@app.route('/view/<int:class_name>')
def view(class_name):
    folder = "static/IDPICTURES/"+str(class_name)
    images = []
    for filename in os.listdir(folder):
        img = os.path.join(folder, filename)
        if img is not None:
            images.append(img)

    return render_template("view_class.html", result=images)


if __name__ == "__main__":
    # decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    # run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
# optional if we want to run in debugging mode
# app.run(debug=True)
