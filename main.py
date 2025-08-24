import cv2
import numpy as np
import streamlit as st
from keras.applications.mobilenet_v2 import (MobileNetV2, preprocess_input, decode_predictions)
from PIL import Image

def load_model():
    model = MobileNetV2(weights = 'imagenet')
    return model

def preprocess_image(image):
    # Convert PIL image to numpy array
    img = np.array(image)
    
    # Ensure the image is in the correct format (RGB)
    if img.shape[-1] == 4:  # RGBA to RGB
        img = img[:, :, :3]
    
    # Resize to 224x224
    img = cv2.resize(img, (224, 224))
    
    # Convert to float32 for preprocessing
    img = img.astype(np.float32)
    
    # Apply MobileNetV2 preprocessing and ensure it's a numpy array
    img = preprocess_input(img)
    
    # Force conversion to numpy array if it's a tensor
    img = np.asarray(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def classify_image(model,image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None
    

def main():
    st.set_page_config(page_title="AI Image Classification", page_icon="🔍", layout="centered")
    st.title("AI Image Classification with MobileNetV2")
    st.write("Upload an image to classify it using AI!")

    # Add loading message
    with st.spinner("Loading AI model... This may take a moment on first run."):
        @st.cache_resource
        def load_cached_model():
            try:
                st.write("Initializing MobileNetV2 model...")
                model = load_model()
                st.success("✅ Model loaded successfully!")
                return model
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                return None
        
        model = load_cached_model()

    if model is None:
        st.error("Failed to load the AI model. Please refresh the page.")
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify image")

        if btn:
            with st.spinner("Classifying..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model,image)

                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")
                else:
                    st.error("Could not classify the image. Please try another image.")

if __name__ == "__main__":
    main()
