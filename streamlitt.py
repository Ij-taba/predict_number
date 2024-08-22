import streamlit as st
from PIL import Image
import numpy as np

# Define the model path and load the model
model_path = "age1_prediction_model.h5"

# Load the model using numpy (if the model is saved as a numpy array)
model = np.load(model_path, allow_pickle=True)

# Streamlit UI
st.markdown(
    """
    <h1 style='color:grey; font-size: 48px;'>Handwritten Number Prediction</h1>
    """,
    unsafe_allow_html=True
)

upload_file = st.file_uploader("Choose picture", type=["jpg", "png"])

if upload_file is not None:
    # Open and preprocess the image
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale and resize using Pillow
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28

    # Convert to numpy array and normalize
    image_array = np.asarray(image)
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model input
    image_array = image_array / 255  # Normalize

    # Predict (Assuming model is a simple classifier in numpy array form)
    pr = model.predict(image_array)
    pr = np.argmax(pr)
    st.write(f"Number is: {pr}")

else:
    st.write("Please upload an image file to display.")
