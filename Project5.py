import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Dataset paths - UPDATE THIS TO YOUR D: DRIVE PATH
DATA_DIR = r"D:\vscode\.vscode\Python\practice\dataset\images"  # Change to your exact path
CLASSES = ['bird', 'drone']

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
@st.cache_data
def load_datasets():
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    valid_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
    return train_dataset, valid_dataset, test_dataset

train_dataset, valid_dataset, test_dataset = load_datasets()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18 and modify final layer
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    return model

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function WITHOUT per-epoch UI spam
def train_model(epochs=10):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Update progress bar only (no epoch logs)
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Epoch {epoch+1}/{epochs} completed')
    
    progress_bar.progress(1.0)
    st.success("✅ Training completed!")

# Validation function
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    return acc

# Dataset info display
col1, col2, col3 = st.columns(3)
col1.metric("Train Images", len(train_dataset))
col2.metric("Valid Images", len(valid_dataset))
col3.metric("Test Images", len(test_dataset))

# Streamlit interface
st.title('🚀 Custom CNN Image Classifier (ResNet18)')
st.markdown("**Dataset connected:** " + DATA_DIR)

if st.button('🎯 Train Model (10 Epochs)', type="primary"):
    with st.spinner('Training in progress...'):
        train_model(epochs=10)

# Show validation accuracy
if len(valid_dataset) > 0:
    val_acc = evaluate(valid_loader)
    st.metric("Validation Accuracy", f"{val_acc*100:.2f}%")

# Image upload for prediction
st.subheader("📸 Predict New Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred = torch.max(probs, 0)
        class_name = CLASSES[pred]
        
        col1, col2 = st.columns(2)
        col1.success(f"**Prediction:** {class_name}")
        col2.success(f"**Confidence:** {conf.item():.2f}")
        
        # Show confidence for both classes
        st.subheader("Confidence Scores")
        conf_df = {"Class": CLASSES, "Confidence": [probs[0].item(), probs[1].item()]}
        st.dataframe(conf_df)
