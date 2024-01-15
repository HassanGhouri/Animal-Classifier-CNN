# Animal-Classifier-CNN

Overview:
Convolutional Neural Network to classify over 60 different species of animals using a website.

Project Description:
Created a Convolutional Neural Network using Tensorflow and Keras to identiy over 60 different species of animals. Used Flask and HTML to create a website where users can upload pictures of animals for the models to identify and print out a perdiction.

To run:
Download and unzip all the files in one folder, and download all dependencies at the top of each file.
Note that the cnnModel.py file will not work as there is no file contaning picutres of animals for the file to use and create a cnn model with. 
There is already a cnn model saved in the modelSave.zip file so the website will still be able to run.

Limitations and Future upgrades:
Due to the complexity of the task, the models preformace is limmited by the size of the dataset. The dataset it was trained on containes over 4000 images of 63 animals. Althoguht this is a fairly large dataset it is not large enough for the model to get a throught understanding of the features invloved in indetifiy each and every animal, which leads to the model sometimes wrongfully idenfiity certain animals. In the future I would like to make updates to this project by using a larger dataset.

Dependencies:
Tensorflow, Keras, numpy, os, Flask
