import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Define the model architecture
class TBModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.densenet121(pretrained=False)
        self.base.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.base(x)

# Load model
@st.cache_resource
def load_model():
    model = TBModel()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🫁 Tuberculosis Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image and the model will predict if TB is present.")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    with st.spinner("Analyzing image..."):
        model = load_model()

        # Preprocess the image
        input_image = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_image)
            prob = torch.sigmoid(output).item()
            predicted = 1 if prob > 0.5 else 0

        # Show result
        st.subheader("Prediction Result")
        if predicted == 1:
            st.error(f"Prediction: **Tuberculosis** detected with {prob:.2f} confidence")
        else:
            st.success(f"Prediction: **Normal** with {(1 - prob):.2f} confidence")
