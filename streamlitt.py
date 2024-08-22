# import streamlit as st
# from PIL import Image
# import numpy as np
#
# import tensorflow
# import cv2
#
# #st.title("Age Prediction system")
# st.markdown(
#     """
#     <h1 style='color:grey; font-size: 48px;'>Handwriten Number Prediction</h1>
#     """,
#     unsafe_allow_html=True
# )
#
# upload_file=st.file_uploader("chose picture",type=["jpg","png"])
# model=tensorflow.models.load_model("age1_prediction_model.h5")
# if upload_file is not  None:
#     image=Image.open(upload_file)
#     st.image(image,caption="uploaded image",use_column_width=True)
#     image=image.convert('L')
#     image=np.asarray(image)
#
#     image=cv2.resize(image,(28,28))
#     image = image.reshape(1, 28, 28, 1)
#     image=image/255
#
#     pr=model.predict(image)
#
#     pr=np.argmax(pr)
#     st.write(f"number is->{pr}")
#
#
#
#
# else:
#     st.write("plz upload image file to display")
#
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib  # For loading scikit-learn models

# Define the model path and load the model
model_path = "handwritten_number_model.pkl"
model = joblib.load(model_path)

# Streamlit UI
st.markdown(
    """
    <h1 style='color:grey; font-size: 48px;'>Handwritten Number Prediction</h1>
    """,
    unsafe_allow_html=True
)

upload_file = st.file_uploader("Choose picture", type=["jpg", "png"])

if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    image = np.asarray(image)
    image = cv2.resize(image, (28, 28))  # Resize to 28x28 if necessary
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    image = image / 255  # Normalize

    # Predict
    pr = model.predict(image)
    pr = np.argmax(pr)
    st.write(f"Number is: {pr}")

else:
    st.write("Please upload an image file to display.")
