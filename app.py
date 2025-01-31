import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Set up the page configuration
st.set_page_config(page_title="Indian Common Bird Detector", layout="wide")

# Device setup (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
@st.cache_resource
def load_model(model_path="./models/checkpoint.pth", num_classes=25):
    model = models.efficientnet_b0(pretrained=False)  # ✅ Fixed: Using EfficientNet-B0
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_classes)  # ✅ Fixed: Match train.py
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded model checkpoint from: {model_path}")
    else:
        print("⚠️ No checkpoint found, loading model from scratch.")
    
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Helper function to predict
def predict(image, model, class_names):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
    return class_names[predicted.item()], confidence

# Load class names
@st.cache_resource
def load_class_names():
    return ['Asian Green Bee Eater', 'Brown Headed Barbet', 'Cattle Egret', 'Common Kingfisher', 'Common Myna', 
            'Common Rosefinch', 'Common Tailorbird', 'Coppersmith Barbet', 'Forest Wagtail', 'Gray Wagtail', 
            'Hoopoe', 'House Crow', 'Indian Grey Hornbill', 'Indian Peacock', 'Indian Pitta', 'Indian Roller', 
            'Jungle Babbler', 'Northern Lapwing', 'Red Wattled Lapwing', 'Ruddy Shelduck', 'Rufous Treepie', 
            'Sarus Crane', 'White Breasted Kingfisher', 'White Breasted Waterhen', 'White Wagtail']

class_names = load_class_names()

# UI Layout
st.title("🦜 Indian Birds Detector")
st.markdown("Upload an image to detect **Indian bird species** using a deep learning model.")

# Upload and display image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Resize image before displaying to prevent taking full height
    image_resized = image.resize((300, 300))  # ✅ Set max size

    # Use columns to center the image and reduce space usage
    col1, col2, col3 = st.columns([1, 2, 1])  # Centering layout
    with col2:
        st.image(image_resized, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict"):
        class_name, confidence = predict(image, model, class_names)
        st.subheader(f"Prediction: **{class_name}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")