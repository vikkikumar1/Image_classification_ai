# 🧠 Image_classification_ai - Animal Image Classification

Image_classification_ai is a deep learning-based image classification web application built using PyTorch, EfficientNet-B0, and Streamlit. The model is fine-tuned on the Animals-10 dataset and can classify uploaded animal images into 10 different categories with high accuracy.

## 🚀 Features

* Upload an animal image
* Real-time image classification
* Fine-tuned EfficientNet-B0 model
* Displays prediction confidence score
* Shows Top-3 predicted classes
* Interactive Streamlit web interface
* High accuracy (~97%)


## 🏗️ Tech Stack

* Python
* PyTorch
* TorchVision
* EfficientNet-B0
* Streamlit
* NumPy
* Pillow
* TIMM


## 📂 Dataset

Animals-10 Dataset

Classes:

* Dog
* Horse
* Elephant
* Butterfly
* Chicken
* Cat
* Cow
* Sheep
* Spider
* Squirrel

## 📊 Model Performance

| Metric     | Score           |
| ---------- | --------------- |
| Accuracy   | ~97%            |
| Model      | EfficientNet-B0 |
| Classes    | 10              |
| Image Size | 224 × 224       |


## 📸 Application Preview

Upload an image and the system will:

* Predict the animal category
* Display confidence score
* Show Top-3 predictions


## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/IMGVision-AI.git
cd IMGVision-AI
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```


## 📜 License

This project is open-source and available under the MIT License.
