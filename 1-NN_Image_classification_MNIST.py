import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms



# Define neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()  # Call the constructor of the parent class (nn.Module)
        
        # Flatten layer to convert 2D images into 1D tensors
        self.flatten = nn.Flatten()
        
        # Fully connected (dense) layer with 28*28 input features and 128 output features
        self.fc1 = nn.Linear(28*28, 128)
        
        # Rectified Linear Unit (ReLU) activation function
        self.relu = nn.ReLU()
        
        # Fully connected (dense) layer with 128 input features and 10 output features
        # (corresponding to the 10 classes of the MNIST digits)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten the input tensor to convert it into a 1D tensor
        x = self.flatten(x)
        
        # Pass the flattened tensor through the first fully connected layer
        x = self.fc1(x)
        
        # Apply the ReLU activation function to introduce non-linearity
        x = self.relu(x)
        
        # Pass the output of the first fully connected layer through the second fully connected layer
        x = self.fc2(x)
        
        # Return the output tensor (logits)
        return x


# Define transforms to apply to the data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to range [-1, 1]
])

# Download and load the training set
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)


# Create data loaders to iterate through the dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

EPOCHS=5

# Training loop
for epoch in range(EPOCHS):  # 5 epochs for demonstration
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Training Accuracy: {(correct_train/total_train) * 100:.2f}%")

# Save the entire model
torch.save(model.state_dict(), 'simple_nn_model_MNIST.pth')
