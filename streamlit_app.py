import streamlit as st
import keras_cv as cv
# import matplotlib.pyplot as plt

model = cv.models.StableDiffusion(img_width=512, img_height=512)

input = st.textarea("Give your prompt here...")

images = model.text_to_image(input, batch_size=3)

def plot_images(images):
  for i in range(len(images)):
    st.image(images[i])
    
