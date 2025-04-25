from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('rice_grain_model.h5')  

class_names = ['Arborio', 'Basmati', 'ipsala', 'Jasmin', 'Karacadag']  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files.get('image')
    if uploaded_file:
        img_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions)) * 100
        predicted_label = class_names[predicted_index]
        prediction_result = f"{predicted_label} ({confidence:.2f}%)"

        return render_template('index.html', prediction=prediction_result)
    
    return render_template('index.html', prediction="No file uploaded!")

if __name__ == '__main__':
    app.run(debug=True)
