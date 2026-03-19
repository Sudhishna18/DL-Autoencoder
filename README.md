# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Image classification is a fundamental task in computer vision where images are categorized into predefined classes. The objective of this project is to develop an image classification model using transfer learning with the pre-trained VGG19 architecture, adapting its learned features to the given dataset, and to evaluate the model’s performance using appropriate metrics.

## Neural Network Model
<img width="1248" height="963" alt="image" src="https://github.com/user-attachments/assets/ee4d8230-1165-4663-8f71-c4b89a56a5ee" />


## DESIGN STEPS
### STEP 1: 

Import required libraries and define image transforms.

### STEP 2: 

Load training and testing datasets using ImageFolder.

### STEP 3: 

Visualize sample images from the dataset.

### STEP 4: 

Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 

Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.

### STEP 6: 

Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.
## PROGRAM

### Name: Sudhishna P

### Register Number: 212224040336

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])
!unzip -qq ./chip_data.zip -d data
# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)

# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()

# Show sample images from the training dataset
show_sample_images(train_dataset)

# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Get the total number of samples in the testing dataset
print(f"Total number of test samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset

first_image, label = test_dataset[0]
print(f"Shape of the first image: {first_image.shape}")
# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model=models.vgg19(weights=VGG19_Weights.DEFAULT)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers
# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader,test_loader,num_epochs=10):
    # Write your code here
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        # Compute validation loss
        # Write your code here
        running_loss = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for images, labels in test_loader:
              images, labels = images.to(device), labels.to(device)
              outputs=model(images)
              loss=criterion(outputs,labels.unsqueeze(1).float())
              val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Sudhishna P")
    print("Register Number:212224040336 ")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_model(model,train_loader,test_loader)

## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels=labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            probs=torch.sigmoid(outputs)
            predicted=(probs>0.5).int()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Name: Sudhishna P")
    print("Register Number: 212224040336")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Name: Sudhishna P")
    print("Register Number: 212224040336")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Evaluate the model
test_model(model,test_loader)

## Step 5: Predict on a Single Image and Display It
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Apply sigmoid to get probability, threshold at 0.5
        prob = torch.sigmoid(output)
        predicted = (prob > 0.5).int().item()


    class_names = class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    print("Name: Sudhishna P")
    print("Register Number:212224040336")
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')

predict_image(model, image_index=55, dataset=test_dataset)

```

### OUTPUT

<img width="717" height="129" alt="image" src="https://github.com/user-attachments/assets/89d74e0f-5746-48c4-9356-a5fd907fe28e" />
<img width="855" height="55" alt="image" src="https://github.com/user-attachments/assets/db417654-1420-4275-bf3a-228bf74fa918" />
<img width="866" height="59" alt="image" src="https://github.com/user-attachments/assets/ec853221-1964-4952-ad61-87d205bd5aa1" />
<img width="517" height="850" alt="image" src="https://github.com/user-attachments/assets/bb667f90-2b28-493b-9c4a-aa93107c79c0" />
<img width="545" height="852" alt="image" src="https://github.com/user-attachments/assets/ab0fb8c1-a84f-4315-bc9d-e9ce5127ffc4" />

## Training Loss, Validation Loss Vs Iteration Plot

<img width="873" height="775" alt="image" src="https://github.com/user-attachments/assets/b91a6d32-088e-4786-a71a-7ad38e1631fe" />

## Confusion Matrix

<img width="875" height="760" alt="image" src="https://github.com/user-attachments/assets/3f6283c4-8706-4838-b884-b8c4f607abe5" />

## Classification Report

<img width="873" height="283" alt="image" src="https://github.com/user-attachments/assets/957f3f8a-c54f-4614-b383-5ae604b0e866" />

### New Sample Data Prediction

<img width="877" height="555" alt="image" src="https://github.com/user-attachments/assets/c9a8a280-8f62-4030-9495-4ff7b943b654" />

## RESULT
The image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
