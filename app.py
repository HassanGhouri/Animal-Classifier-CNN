from flask import Flask, render_template, request
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# Dictionary of classes of animals
classes = {0: 'antelope', 1: 'badger', 2: 'bear', 3: 'bee', 4: 'bison', 5: 'boar', 6: 'chimpanzee', 7: 'cockroach',
           8: 'cow', 9: 'crow', 10: 'deer', 11: 'dog', 12: 'dolphin', 13: 'donkey', 14: 'eagle', 15: 'elephant',
           16: 'flamingo', 17: 'fly', 18: 'fox', 19: 'goat', 20: 'goldfish', 21: 'gorilla', 22: 'grasshopper',
           23: 'hare', 24: 'hedgehog', 25: 'hippopotamus', 26: 'hornbill', 27: 'horse', 28: 'hummingbird', 29: 'hyena',
           30: 'jellyfish', 31: 'koala', 32: 'ladybugs', 33: 'leopard', 34: 'lion', 35: 'mosquito', 36: 'moth',
           37: 'mouse', 38: 'okapi', 39: 'orangutan', 40: 'owl', 41: 'panda', 42: 'penguin', 43: 'pigeon', 44: 'possum',
           45: 'raccoon', 46: 'rat', 47: 'reindeer', 48: 'rhinoceros', 49: 'sandpiper', 50: 'seahorse', 51: 'seal',
           52: 'shark', 53: 'sheep', 54: 'snake', 55: 'sparrow', 56: 'squirrel', 57: 'starfish', 58: 'tiger',
           59: 'turkey', 60: 'turtle', 61: 'whale', 62: 'wolf', 63: 'zebra'}

# Load the trained CNN modelSave
cnn = tf.keras.models.load_model('modelSave')

# Ensure the 'static/uploads/' directory exists
upload_dir = "static/uploads/"
os.makedirs(upload_dir, exist_ok=True)


# Func to preprocess and make predictions
def predict_animal(image_path):
    """
    Use ML modelSave to predict animal in image.
    :param image_path: Path to image
    :return: ml modelSave prediction
    """
    test_img = image.load_img(image_path, target_size=(64, 64))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    result = cnn.predict(test_img)
    return result


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Take uploaded image, use ML modelSave to predict animal, display perdition alongside uploaded image.
    :return: None
    """
    if request.method == 'POST':

        # Handle uploaded file
        file = request.files['file']

        if file:
            # Save the uploaded file to the uploads directory
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)

            # Get CNN prediction
            result = predict_animal(file_path)
            predicted_class_index = np.argmax(result)
            predicted_class = classes[predicted_class_index]

            # Render the template with the uploaded image and CNN prediction
            return render_template('index.html', file_path='uploads/' + file.filename, predicted_class=predicted_class)

    return render_template('index.html', file_path=None, predicted_class=None)


if __name__ == '__main__':
    app.run(debug=True, port=8080)
