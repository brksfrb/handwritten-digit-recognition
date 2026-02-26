import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DigitNet

# ================================
# 1. DATA AUGMENTATION
# ================================
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),

    transforms.RandomAffine(
        degrees=35,
        translate=(0.25, 0.25),
        scale=(0.7, 1.4),
        shear=15
    ),

    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ================================
# 2. LOAD DATASET
# ================================
dataset = datasets.ImageFolder("../data/", transform=train_transform)

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Apply clean transform to validation data
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ================================
# 3. MODEL
# ================================
model = DigitNet()


# ================================
# 4. TRAINING SETUP
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ================================
# 5. TRAINING LOOP
# ================================
epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")


# ================================
# 6. SAVE MODEL
# ================================
if not os.path.exists('weights'):
    os.mkdir('weights')
torch.save(model.state_dict(), "weights/digit_model.pth")

print("Model saved as digit_model.pth!")

# ================================
# 7. CONFUSION MATRIX
# ================================
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Collect all true labels and predictions
all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("Validation Confusion Matrix")

# Save and show
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/confusion_matrix.png")
plt.show()
