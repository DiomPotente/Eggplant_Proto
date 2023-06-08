import streamlit as st
import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def classify_image(img):
    model = VGG16(weights='imagenet')
    img = img.resize((224, 224))  # Resize the image to match the input size of VGG16
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    predicted_class = decode_predictions(preds, top=1)[0][0][1]
    probability = decode_predictions(preds, top=1)[0][0][2]
    return predicted_class, probability


def main():
    st.title("Image Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify"):
            predicted_class, probability = classify_image(img)
            st.success(f"Predicted Class: {predicted_class}")
            st.success(f"Probability: {probability}")

if __name__ == "__main__":
    main()
