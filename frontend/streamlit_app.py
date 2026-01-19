import io
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Ghibli Converter", layout="wide")
st.title("Image → Ghibli (CycleGAN)")

API_BASE = st.text_input("Backend API URL", "http://localhost:8000")

def call_api(img_bytes: bytes, filename="image.jpg"):
    files = {"file": (filename, img_bytes, "image/jpeg")}
    r = requests.post(f"{API_BASE}/stylize", files=files, timeout=120)
    r.raise_for_status()
    return r.content

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])

if uploaded and st.button("Convert"):
    img = Image.open(uploaded).convert("RGB")
    out_bytes = call_api(uploaded.getvalue(), filename=uploaded.name)
    out = Image.open(io.BytesIO(out_bytes)).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original", use_container_width=True)
    with col2:
        st.image(out, caption="Ghibli", use_container_width=True)
