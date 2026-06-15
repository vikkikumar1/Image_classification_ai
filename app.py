import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# =====================================================

# PAGE CONFIG

# =====================================================

st.set_page_config(
page_title="IMGVision AI",
page_icon="🧠",
layout="wide"
)

# =====================================================

# CUSTOM CSS

# =====================================================

st.markdown("""

<style>

.main {
    padding-top: 1rem;
}

.title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#4CAF50;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:gray;
    margin-bottom:20px;
}

.prediction-box{
    padding:20px;
    border-radius:12px;
    background-color:#262730;
    text-align:center;
    font-size:25px;
    font-weight:bold;
}

</style>

""", unsafe_allow_html=True)

# =====================================================

# CLASS NAMES

# =====================================================

class_mapping = {
0: "cane",
1: "cavallo",
2: "elefante",
3: "farfalla",
4: "gallina",
5: "gatto",
6: "mucca",
7: "pecora",
8: "ragno",
9: "scoiattolo"
}

english_names = {
"cane": "Dog",
"cavallo": "Horse",
"elefante": "Elephant",
"farfalla": "Butterfly",
"gallina": "Chicken",
"gatto": "Cat",
"mucca": "Cow",
"pecora": "Sheep",
"ragno": "Spider",
"scoiattolo": "Squirrel"
}

# =====================================================

# LOAD MODEL

# =====================================================

@st.cache_resource
def load_model():
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=False
    )

    model.classifier = nn.Linear(
        model.classifier.in_features,
        10
    )

    checkpoint = torch.load(
        "best_efficientnet.pth",
        map_location=torch.device("cpu")
    )

    # Supports both state_dict and checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model

try:
    model = load_model()

except Exception as e:
    st.error(f"❌ Model Loading Error: {e}")
    st.stop()

# =====================================================

# IMAGE TRANSFORM

# =====================================================

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)
])

# =====================================================

# PREDICTION FUNCTION

# =====================================================

def predict(image):
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        confidence, predicted = torch.max(
            probabilities,
            dim=0
        )

    predicted_idx = predicted.item()

    italian_label = class_mapping[predicted_idx]
    english_label = english_names.get(italian_label, italian_label)

    return (
        english_label,
        confidence.item() * 100,
        probabilities.numpy()
    )


# =====================================================

# HEADER

# =====================================================

st.markdown(
'<div class="title">🧠 IMGVision AI</div>',
unsafe_allow_html=True
)

st.markdown(
'<div class="subtitle">Fine-Tuned EfficientNet-B0 Image Classification System</div>',
unsafe_allow_html=True
)

st.divider()

# =====================================================

# SIDEBAR

# =====================================================

with st.sidebar:
    st.header("📌 Project Information")

    st.write("""

    **Model:** EfficientNet-B0

    **Dataset:** Animals-10 (raw-img)

    **Classes:** 10

    **Framework:** PyTorch

    **Frontend:** Streamlit

    **Accuracy:** ~97%
    """)

    st.success("✅ Model Loaded Successfully")


# =====================================================

# MAIN LAYOUT

# =====================================================

left_col, right_col = st.columns([1, 1])

# =====================================================

# IMAGE UPLOAD

# =====================================================

with left_col:
    st.subheader("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
        )

# =====================================================

# PREDICTION PANEL

# =====================================================

with right_col:
    st.subheader("🔍 Prediction")

    if uploaded_file is not None:
        if st.button("🚀 Classify Image"):
            with st.spinner("Analyzing image..."):
                label, confidence, probs = predict(image)

            st.markdown(
                f"""
                <div class="prediction-box">
                Prediction: {label.upper()}
                <br><br>
                Confidence: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            if confidence >= 90:
                st.success(f"High Confidence Prediction ({confidence:.2f}%)")
            elif confidence >= 70:
                st.warning(f"Moderate Confidence Prediction ({confidence:.2f}%)")
            else:
                st.error(f"Low Confidence Prediction ({confidence:.2f}%)")

            st.subheader("📊 Top 3 Predictions")

            top3_idx = np.argsort(probs)[::-1][:3]

            for idx in top3_idx:
                italian_label = class_mapping[int(idx)]
                english_label = english_names.get(italian_label, italian_label)
                score = probs[int(idx)] * 100

                st.write(f"**{english_label}** : {score:.2f}%")
                st.progress(float(probs[int(idx)]))

# =====================================================

# FOOTER

# =====================================================

st.divider()

st.markdown("""

<center>

### 🚀 IMGVision AI

Deep Learning Based Multi-Class Animal Image Classification

Built with ❤️ using PyTorch, EfficientNet-B0 and Streamlit

Developed by Vikki Kumar

</center>
""", unsafe_allow_html=True)