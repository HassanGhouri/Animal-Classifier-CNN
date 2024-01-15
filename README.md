# Animal-Classifier-CNN

## Overview:
Convolutional Neural Network to classify over 60 different species of animals using a website.

## Project Description:
Created a Convolutional Neural Network using Tensorflow and Keras to identiy over 60 different species of animals. Utilized Flask and HTML to create a website where users can upload pictures of animals for the models to identify and print out a perdiction.

## How To Run:
1. Download and unzip all the files into one folder.
2. Download dependencies listed at the top of each file.
3. Note: `cnnModel.py` will not work as there is no file containing pictures of animals for the model to use. A pre-trained model is available in the `modelSave.zip` file.

## Demo:
A Demo video has been inluded, check "Demo".

Limitations and Future upgrades:
Due to the complexity of the task, the models preformace is limmited by the size of the dataset. The dataset it was trained on containes over 4000 images of 63 animals. Althoguht this is a fairly large dataset it is not large enough for the model to get a throught understanding of the features invloved in indetifiy each and every animal, which leads to the model sometimes wrongfully idenfiity certain animals. In the future I would like to make updates to this project by using a larger dataset.

## Dependencies:
- Tensorflow
- Keras
- Numpy
- os
- Flask
