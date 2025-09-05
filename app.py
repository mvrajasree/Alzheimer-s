import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import spacy
import subprocess

# Function to download the SpaCy model if not already installed
@st.cache_resource
def download_spacy_model():
    model_name = "en_core_web_sm"
    try:
        # Check if the model is already installed
        spacy.load(model_name)
        print(f"Model '{model_name}' already installed.")
    except OSError:
        print(f"Model '{model_name}' not found. Downloading...")
        # Use subprocess to run the download command
        subprocess.run(["python", "-m", "spacy", "download", model_name])
        print(f"Model '{model_name}' downloaded successfully.")

# Call the function at the start of your app
download_spacy_model()

# Now you can safely load the model
nlp = spacy.load("en_core_web_sm")

# --- Your app code starts here ---
st.title("Alzheimer's Disease Prediction")
# ... rest of your code

# Load the saved Keras model
model = tf.keras.models.load_model('alzheimer_classification_model.keras')

# Define class names
class_names = ['MildDemented', 'NonDemented', 'ModerateDemented', 'VeryMildDemented']


st.write('Upload an MRI image to classify the stage of Alzheimer\'s disease.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    # Make a prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Get the predicted class and confidence
    predicted_class_index = np.argmax(score)
    predicted_class_name = class_names[predicted_class_index]
    confidence = 100 * np.max(score)

    # Display the results
    st.write(f"Prediction: {predicted_class_name}")
    st.write(f"Confidence: {confidence:.2f}%")
