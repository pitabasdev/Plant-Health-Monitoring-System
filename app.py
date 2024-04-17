from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('C:/Users/user/Documents/plant/model.h5')

def preprocess_image(image):
    img = image.resize((64, 64))
    img = np.array(img) / 255.0
    return img.reshape(1, 64, 64, 3)

@app.route('/')
def index():
    return render_template('index.html', result=None)  # Pass result=None initially

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image found!"
    
    file = request.files['image']
    if file.filename == '':
        return "No image selected!"

    img = Image.open(file)
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)

    result = "Healthy" if prediction[0][0] > 0.5 else "Unhealthy"
    return render_template('index.html', result=result)  

if __name__ == '__main__':
    app.run(debug=True)
