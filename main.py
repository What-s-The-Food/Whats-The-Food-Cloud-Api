# Import library
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
from io import BytesIO

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# Flask Object 
app = Flask(__name__)

ALLOW_EXTENSION = {'jpg','jpeg'}

# Load the Machine Learning Model
model = tf.keras.models.load_model('model/CapstoneModel90val.h5',custom_objects={'KerasLayer':hub.KerasLayer})

# Function to allow files format
def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSION

# Function to read the images
def read_image(image):
    img = Image.open(BytesIO(image))
    img = img.resize((160, 160), Image.ANTIALIAS)
    img = img_to_array(img)
    img /= 255
    img = np.expand_dims(img, axis=0)
    return img

# Server test function
@app.route('/') 
def index():
    return "What's the Food Apps"

# Route Predict Images
@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']

    if image and allow_file(image.filename):
        image = image.read()
        img = read_image(image)
        results = model.predict(img)
        max_value = np.max(results)

        if max_value <= 0.08:
            resp = jsonify({'message': 'Image can not be predicted'})
            resp.status_code = 400
            return resp 
        elif max_value > 0.08:
            result = np.argmax(results, axis = 1)

            if result == 0:
                Calories = "246 gram"
                Carbohydrate = "1.8 gram"
                Protein = "10 gram"
                Fat = "12 gram"
                Name = "Ayam Goreng" +" ({:.0%})".format(max_value)
            elif result == 1:
                Calories = "57gram"
                Carbohydrate = "2.12 gram"
                Protein = "3.46 gram"
                Fat = "3.69 gram"
                Name = "Bakso" +" ({:.0%})".format(max_value)
            elif result == 2:
                Calories = "290 gram"
                Carbohydrate = "29.12 gram"
                Protein = "10.28 gram"
                Fat = "14.92 gram"
                Name = "Batagor" +" ({:.0%})".format(max_value)
            elif result == 3:
                Calories = "50 gram" 
                Carbohydrate = "11 gram"
                Protein = "10.28 gram"
                Fat = "14.92 gram"
                Name = "Bubur" +" ({:.0%})".format(max_value)
            elif result == 4:
                Calories = "295 gram"
                Carbohydrate = "24 gram"
                Protein = "17 gram"
                Fat = "144 gram"
                Name = "Burger" +" ({:.0%})".format(max_value)
            elif result == 5:
                Calories = "67 gram"
                Carbohydrate = "9.67 gram"
                Protein = "1.58 gram"
                Fat = "2.59 gram"
                Name = "Capcay" +" ({:.0%})".format(max_value)
            elif result == 6:
                Calories = "224 gram"
                Carbohydrate = "21.66 gram"
                Protein = "8.76 gram"
                Fat = "11.08 gram"
                Name = "Crepes" +" ({:.0%})".format(max_value)
            elif result == 7:
                Calories = "125 gram"
                Carbohydrate = "9.9 gram"
                Protein = "15.13 gram"
                Fat = "2.17 gram"
                Name = "Cumi Goreng Tepung" +" ({:.0%})".format(max_value)
            elif result == 8:
                Calories = "131 gram"
                Carbohydrate = "3.61 gram"
                Protein = "7.55 gram"
                Fat = "9.71 gram"
                Name = "Fu Yung Hai" +" ({:.0%})".format(max_value)
            elif result == 9:
                Calories = "127 gram"
                Carbohydrate = "22.54 gram"
                Protein = "1.82 gram"
                Fat = "4.71 gram"
                Name = "Gado-gado" +" ({:.0%})".format(max_value)
            elif result == 10:
                Calories = "127 gram"
                Carbohydrate = "22.54 gram"
                Protein = "1.82 gram"
                Fat = "4.71 gram"
                Name = "Gudeg" +" ({:.0%})".format(max_value)
            elif result == 11:
                Calories = "142 gram"
                Carbohydrate = "0.37 gram"
                Protein = "24.79 gram"
                Fat = "3.89 gram"
                Name = "Ikan Bakar" +" ({:.0%})".format(max_value)
            elif result == 12:
                Calories = "185.4 gram"
                Carbohydrate = "11 gram"                
                Protein = "12.1 gram"
                Fat = "10 gram"
                Name = "Kebab" +" ({:.0%})".format(max_value)
            elif result == 13:
                Calories = "468 gram"
                Carbohydrate = "7.8 gram"
                Protein = "22.6 gram"
                Fat = "7.9 gram"
                Name = "Rendang" +" ({:.0%})".format(max_value)
            elif result == 14:
                Calories = "34 gram"
                Carbohydrate = "0.37 gram"
                Protein = "2.93 gram"
                Fat = "2.22 gram"
                Name = "Sate" +" ({:.0%})".format(max_value)
            return jsonify({'Name' : Name, 'Calories' : Calories, 'Carbohydrate' : Carbohydrate, 'Protein' : Protein, 'Fat' : Fat})
        else:
            res = jsonify({'message': 'Image extension is not allowed'})
            res.code_status = 400
            return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



