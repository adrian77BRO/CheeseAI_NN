from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Cargar modelo entrenado
model = tf.keras.models.load_model("mejor_modelo.h5")

class_names = ["americano",
               "azul",
               "botanero",
               "cheddar",
               "chihuahua",
               "chile",
               "cotija",
               "cottage",
               "fresco",
               "panela",
               "parmesano",
               "philadelphia",
               "quesillo"]

# Permitir conexión desde app móvil y web
CORS(app)

@app.route("/predict", methods=["POST"])
def predict_image():
    file = request.files['file']
    image = Image.open(file.stream).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return jsonify({"className": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)