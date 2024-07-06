import streamlit as st
import keras_cv as cv
import numpy as np

model = cv.models.StableDiffusion(img_width=512, img_height=512)

st.title("Stable Diffusion Image Generator")
st.write("Generate images from text prompts using Stable Diffusion.")

prompt = st.textarea("Enter your prompt here:", placeholder="Type something...")

num_images = st.slider("Number of images to generate:", min_value=1, max_value=10, value=3)

if st.button("Generate Images"):
    if prompt.strip() == "":
        st.error("Please enter a prompt to generate images.")
    else:
        st.write("Generating images... Please wait.")
        try:
            images = model.text_to_image(prompt, batch_size=num_images)
            for i in range(len(images)):
                st.image(images[i])
        except Exception as e:
            st.error(f"An error occurred: {e}")

if st.button("Clear Prompt"):
    prompt = ""
    st.experimental_rerun()
