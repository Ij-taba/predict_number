import streamlit as st
from PIL import Image
import numpy as np

# Define a simple model architecture
class SimpleModel:
    def __init__(self):
        # Initialize model layers
        self.weights = {}

    def predict(self, x):
        # Simple forward pass (without activation functions)
        if 'weights1' not in self.weights or 'weights2' not in self.weights:
            raise ValueError("Weights not properly initialized")
        x = np.dot(x, self.weights['weights1'])
        x = np.dot(x, self.weights['weights2'])
        return x

    def set_weights(self, weights):
        # Set the model's weights from loaded weights
        self.weights = weights

# Create the model architecture
model = SimpleModel()

# Load the saved weights from .npz
weights_path = "models_weights.npz"
weights_dict = np.load(weights_path)

# Convert loaded weights to a dictionary format
weights = {key: weights_dict[key] for key in weights_dict.files}

# Set the model weights
model.set_weights(weights)

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
    image_array = image_array.reshape(1, 28 * 28)  # Flatten for simple model
    image_array = image_array / 255  # Normalize

    # Predict using the model with loaded weights
    try:
        pr = model.predict(image_array)
        pr = np.argmax(pr)
        st.write(f"Number is: {pr}")
    except ValueError as e:
        st.write(f"Error: {e}")

else:
    st.write("Please upload an image file to display.")
