# Imports
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Preprocessing Training Set (Applying transformation (Image augmentations) to avoid over-fitting)
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Importing training set
training_set = train_datagen.flow_from_directory('animals/training',
                                                 target_size=(128, 128),
                                                 batch_size=32,
                                                 class_mode='categorical')

# Preprocessing and importing the test set
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory("animals/test_set",
                                            target_size=(128, 128),
                                            batch_size=32,
                                            class_mode='categorical')

# Building the CNN
cnn = tf.keras.models.Sequential()

# 1) Adding convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[128, 128, 3]))

# 2) Adding the Pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 3) Adding second convolutional layer with another layer of max pooling
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# 4 Flattening (Flattening results into one a dimensional vector to become input for NN)
cnn.add(tf.keras.layers.Flatten())

# 5 Full Connection
cnn.add(tf.keras.layers.Dense(units=256, activation="relu"))

# Adding second hidden layer
cnn.add(tf.keras.layers.Dense(units=256, activation="relu"))

# 6Adding output layer
cnn.add(tf.keras.layers.Dense(units=64, activation="sigmoid"))

# Compiling CNN
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Training CNN on training set
cnn.fit(x=training_set, validation_data=test_set, epochs=32)

# Saving the modelSave
cnn.save("modelSave")
