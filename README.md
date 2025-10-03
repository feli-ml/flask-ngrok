🚀 Flask-Ngrok: API para Predicción con Modelo KNN
Este proyecto crea una API REST con Flask y la expone públicamente usando ngrok. La API permite realizar predicciones con un modelo KNN previamente entrenado, ideal para entornos como Google Colab donde no se puede abrir puertos directamente.

📦 Requisitos
  - Python 3.8+
  - Cuenta en ngrok.com
  - Archivos del modelo: knn_model.pkl y knn_scaler.pkl
  - Archivo .env con tu token de ngrok:

        NGROK_TOKEN=tu_token_aquí

🧰 Instalación
Instala las dependencias necesarias:

    pip install flask pyngrok python-dotenv joblib numpy flask-cors

⚙️ Estructura del Proyecto

    flask-ngrok/
    ├── app.py
    ├── knn_model.pkl
    ├── knn_scaler.pkl
    ├── .env
    └── README.md

🌐 ¿Qué hace este proyecto?
  - Crea un túnel público con ngrok en el puerto 5000.
  - Carga un modelo KNN y su escalador desde archivos .pkl.
  - Expone dos rutas:
    - /: Página de bienvenida.
    - /predict: Ruta POST para enviar datos y recibir una predicción.
