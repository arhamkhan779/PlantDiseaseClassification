import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("densenet_model.keras")

# Define label mapping
label_map = {
    0: "Aloe Vera",
    1: "Areca Palm",
    2: "Pothos"
}

def process_single_image(image):
    """
    Preprocess image for model prediction.
    """
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.reshape(img_array, (1, 224, 224, 3))
    return img_array

# Streamlit UI
st.set_page_config(page_title="🌱 Plant Disease Classifier", layout="centered")

st.title("🌿 Plant Disease Classification")
st.markdown("""
Upload an image of a plant leaf and this AI model will classify it into one of the following classes:
- **Aloe Vera**
- **Areca Palm**
- **Pothos**
""")

uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Classifying..."):
        image = Image.open(uploaded_file)
        processed_image = process_single_image(image)
        prediction = model.predict(processed_image)
        predicted_label = label_map[np.argmax(prediction)]

    st.success(f"✅ **Prediction:** {predicted_label}")
    st.bar_chart(prediction[0])

st.markdown("---")
st.caption("Made with ❤️ using TensorFlow and Streamlit")
