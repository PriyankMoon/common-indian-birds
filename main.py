import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import random

# **Configuration for the model**
model_name = "efficientnet_b0"  # Faster and more accurate than ResNet50
num_classes = 25  # Number of bird species
batch_size = 256  # Increased for faster training
learning_rate = 0.0005  # Slightly reduced for better convergence
epochs = 15  # Reduced from 20 to save time
max_images_per_class = 400  # **Change to 800 for better accuracy**

# **Directories for training and testing data**
train_dir = "./preprocessed-dataset/train"
test_dir = "./preprocessed-dataset/test"

# **Define transformations for image preprocessing**
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# **Load full datasets**
full_train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train"])
full_test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms["test"])

# **Function to select a limited number of images per class**
def get_limited_dataset(full_dataset, max_images_per_class):
    class_to_indices = {class_idx: [] for class_idx in range(len(full_dataset.classes))}
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_to_indices[label].append(idx)

    selected_indices = []
    for indices in class_to_indices.values():
        selected_indices.extend(random.sample(indices, min(len(indices), max_images_per_class)))

    return Subset(full_dataset, selected_indices)

# **Create limited datasets**
train_dataset = get_limited_dataset(full_train_dataset, max_images_per_class)
test_dataset = get_limited_dataset(full_test_dataset, max_images_per_class)

# **Load Data into DataLoader**
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# **Load a pretrained EfficientNet and modify for our dataset**
model = models.efficientnet_b0(pretrained=True)  # Using pretrained weights from ImageNet
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_classes)  # Modify for 25 bird classes

# **Define loss function and optimizer**
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# **Move model to GPU if available**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# **Ensure 'models' folder exists**
model_save_dir = "./models"
os.makedirs(model_save_dir, exist_ok=True)

# **Load checkpoint if available**
def load_checkpoint():
    checkpoint_path = os.path.join(model_save_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("✅ Checkpoint found! Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        return epoch_start
    else:
        print("🚀 No checkpoint found. Starting from scratch!")
        return 0

# **Save checkpoint**
def save_checkpoint(epoch):
    checkpoint_path = os.path.join(model_save_dir, "checkpoint.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"💾 Checkpoint saved at epoch {epoch}")

# **Training loop with progress tracking**
def train_model():
    start_epoch = load_checkpoint()

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        print(f"\n🔥 Starting epoch {epoch + 1}/{epochs}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress = (batch_idx + 1) / num_batches * 100
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{num_batches} - Loss: {running_loss / (batch_idx + 1):.4f}, Progress: {progress:.2f}%")

        save_checkpoint(epoch + 1)

    print("✅ Training complete!")

# **Evaluation loop**
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# **Run training and evaluation**
if __name__ == "__main__":
    train_model()
    evaluate_model()
