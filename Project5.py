import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Configuration
DATA_DIR = r"D:\vscode\.vscode\Python\practice\dataset\images"
CLASSES = ['bird', 'drone']
BATCH_SIZE = 64
EPOCHS = 5

# Detect device (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_data(ttl=300)
def load_datasets():
    try:
        train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
        valid_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=transform)
        return train_ds, valid_ds
    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        return None, None

train_dataset, valid_dataset = load_datasets()

if train_dataset:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=DEVICE.type=='cuda')
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=DEVICE.type=='cuda')

    @st.cache_resource
    def load_model():
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(CLASSES))
        return model.to(DEVICE)

    model = load_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train_model(epochs=EPOCHS):
        model.train()
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_batches = len(train_loader) * epochs

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                current_batch = epoch * len(train_loader) + batch_idx + 1
                progress_bar.progress(current_batch / total_batches)

            status_text.text(f"Epoch {epoch + 1} completed")

        progress_bar.progress(1.0)
        status_text.text("Training complete!")
        st.success("🎉 Training finished!")

    def evaluate(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    # === Streamlit UI ===

    # Header with Dataset and Device info
    st.title("🐦 Drone & Bird Classifier")
    st.markdown(
        f"**Dataset Path:** `{DATA_DIR}`\n\n"
        f"**Device:** {'🟢 CUDA GPU' if DEVICE.type == 'cuda' else '🔴 CPU'}\n\n"
        f"**Train samples:** {len(train_dataset)} | **Validation samples:** {len(valid_dataset)}"
    )
    st.markdown("---")

    # Training and validation controls
    st.header("⚙️ Training Controls")
    col1, col2 = st.columns([2,1])
    with col1:
        epochs_input = st.number_input("Set epochs", min_value=1, max_value=50, value=EPOCHS, step=1)
    with col2:
        if st.button("🚀 Start Training"):
            with st.spinner("Training in progress..."):
                train_model(epochs=epochs_input)

    # Show validation accuracy button
    if st.button("📊 Show Validation Accuracy"):
        val_acc = evaluate(valid_loader)
        st.success(f"Validation Accuracy: {val_acc*100:.2f}%")
        st.balloons()

    st.markdown("---")

    # Image upload and prediction
    st.header("📸 Predict On Image")
    uploaded_file = st.file_uploader("Upload image file (jpg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred = torch.max(probs, 0)

        st.markdown(f"### Prediction: **{CLASSES[pred]}**")
        st.markdown(f"### Confidence: {conf.item():.2%}")

        st.subheader("Confidence Scores")
        conf_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        st.bar_chart(conf_dict)

else:
    st.error(f"Could not load dataset from: {DATA_DIR}")
    st.info("Please ensure you have train/valid folders with class subfolders inside your dataset directory.")
