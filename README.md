# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

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

### Name:Saron Xavier A
### Register Number:212223230197

```python
# Load Pretrained Model and Modify for Transfer Learning
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes
model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer
criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train_model(model, train_loader,test_loader,num_epochs=100):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss=0.0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())

            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        train_losses.append(running_loss/len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss=0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            outputs=model(images)
            loss=criterion(outputs,labels.unsqueeze(1).float())
            val_loss+=loss.item()
        val_losses.append(val_loss/len(test_loader))
        model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
        
    # Plot training and validation loss
    print("Name:Saron Xavier A")
    print("Register Number:212223230197")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Train the model
train_model(model,train_loader,test_loader,num_epochs=10)
```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="822" height="746" alt="image" src="https://github.com/user-attachments/assets/3331e2c7-d9cc-4e42-a0fb-934abb008fc7" />


## Confusion Matrix
<img width="768" height="607" alt="image" src="https://github.com/user-attachments/assets/f161fc0b-1c61-4f94-bf72-cebfe7f76968" />


## Classification Report
<img width="512" height="203" alt="image" src="https://github.com/user-attachments/assets/06f8f9c5-b393-4d37-91b5-9c7335f34296" />


### New Sample Data Prediction
<img width="1170" height="453" alt="image" src="https://github.com/user-attachments/assets/c412373b-08b3-46fb-9272-ff18b1b25e6c" />

## RESULT
Thus, the image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.

