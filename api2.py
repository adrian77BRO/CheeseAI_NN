from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Cargar el modelo
model = tf.keras.models.load_model('mejor_modelo.h5')

# Preprocesar la imagen
def preprocess_image(image):
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    img = Image.open(io.BytesIO(image))
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión para batch
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    image = file.read()
    processed_image = preprocess_image(image)
    
    # Realizar predicción
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Obtener nombre de la clase
    class_names = ['Clase1', 'Clase2', 'Clase3']  # Reemplaza con tus clases
    predicted_class_name = class_names[predicted_class]

    return jsonify({'predicted_class': predicted_class_name}), 200

if __name__ == '__main__':
    app.run(debug=True)