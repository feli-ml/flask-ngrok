# 1. Importar las librerías necesarias
from flask import Flask, request, jsonify
from pyngrok import ngrok
import os
import joblib
import numpy as np
from dotenv import load_dotenv
from flask_cors import CORS

# 1 1/2. Configurar Dotenv
# Asegúrate de que tu archivo .env con el token de ngrok
# esté en la misma carpeta que este script o en /content/.
load_dotenv()

# 2. Configurar tu token de autenticación de ngrok
# Se necesita una cuenta en ngrok.com para obtener un token.
# Lo ideal es guardar este token en un archivo .env.
ngrok_token = os.getenv("NGROK_TOKEN")
if not ngrok_token:
    print("Error: No se encontró el token de ngrok. Asegúrate de tener el archivo .env con 'NGROK_TOKEN=tu_token'.")
else:
    ngrok.set_auth_token(ngrok_token)
    print("Token de ngrok configurado exitosamente.")

# 3. Crear el túnel ngrok
# El túnel se crea en el puerto 5000, que es donde correrá Flask.
public_url = ngrok.connect(5000)
print("URL pública del túnel:", public_url)

# 4. Cargar el modelo y el escalador
# Es crítico cargar ambos, ya que los datos de entrada deben ser escalados
# de la misma manera que los datos de entrenamiento.
try:
    model = joblib.load("/content/knn_model.pkl")
    scaler = joblib.load("/content/knn_scaler.pkl")
    print("Modelo y escalador cargados exitosamente.")
except FileNotFoundError:
    print("Error: Asegúrate de que los archivos 'knn_model.pkl' y 'knn_scaler.pkl' estén en la carpeta /content.")
    model = None
    scaler = None

# 5. Crear y ejecutar tu aplicación Flask
app = Flask(__name__)
CORS(app)  # Permite todas las solicitudes

@app.route("/")
def home():
    return "<h1>¡Hola desde Google Colab con ngrok!</h1><p>Visita /predict para hacer predicciones.</p>"

@app.route("/predict", methods=["POST"])
def predict():
    # Verificar si el modelo y el escalador se cargaron correctamente
    if model is None or scaler is None:
        return jsonify({'error': 'Modelo o escalador no disponibles.'}), 500

    try:
        # Obtener los datos del cuerpo de la solicitud JSON
        data = request.get_json()
        if not data or "features" not in data:
            return jsonify({'error': 'Datos de entrada no válidos. Se esperaba un JSON con la clave "features".'}), 400

        # Convertir la lista de características a un array de NumPy
        features = np.array(data["features"]).reshape(1, -1)

        # APLICAR EL ESCALADO A LOS DATOS DE ENTRADA
        scaled_features = scaler.transform(features)

        # Realizar la predicción con los datos escalados
        prediction = model.predict(scaled_features)

        # Devolver el resultado de la predicción
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        # Manejar cualquier error durante la predicción
        return jsonify({'error': f"Ocurrió un error en la predicción: {e}"}), 500

if __name__ == "__main__":
    app.run()
