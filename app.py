import streamlit as st # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignore
import numpy as np
import cv2 # type: ignore
from tensorflow.keras.models import load_model # type: ignore

model = load_model("alphabet_model.h5")
# Load model


st.title("✍️ Handwritten Alphabet Recognition")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):

    if canvas_result.image_data is not None:

        img = canvas_result.image_data

        # Convert to grayscale
        img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2GRAY)

        # 🔥 IMPORTANT — invert colors
        img = cv2.bitwise_not(img)

        # Resize to 28x28
        img = cv2.resize(img, (28, 28))

        # Normalize
        img = img / 255.0

        # Add batch & channel dimension
        img = img.reshape(1, 28, 28, 1)

        # Predict
        pred = model.predict(img)
        letter = chr(np.argmax(pred) + 65)

        st.success(f"Predicted Letter: {letter}")
