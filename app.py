import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# 1. Page Configuration
st.set_page_config(page_title="AI Image Captioner", page_icon="📸")
st.title("📸 Real-time Image Captioning")
st.markdown("Upload an image or provide a link to see what the AI thinks!")

# 2. Load the AI Model (Cached so it stays fast)
@st.cache_resource
def load_model():
    # We use the 'base' model for speed and efficiency
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# 3. Sidebar for Input Selection
st.sidebar.header("Input Options")
option = st.sidebar.radio("How to provide the image?", ("Upload File", "Image URL"))

image = None

# 4. Handle Image Upload or URL
if option == "Upload File":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')

else:
    url = st.text_input("Paste Image URL (e.g., https://example.com/photo.jpg)")
    if url:
        try:
            response = requests.get(url, stream=True)
            image = Image.open(response.raw).convert('RGB')
        except Exception as e:
            st.error("Error: Could not load image. Make sure the link is direct to an image file.")

# 5. Generate and Display Caption
if image:
    # Show the image to the user
    st.image(image, caption="Your Input Image", use_container_width=True)
    
    with st.spinner('🤖 AI is thinking...'):
        # Prepare the image for the model
        inputs = processor(image, return_tensors="pt")
        
        # Generate the text
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Output the result
        st.success("### Generated Caption:")
        st.write(f"**{caption.capitalize()}**")
