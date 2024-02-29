import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Define neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the testing data
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate the model
model = SimpleNN()

# Load the model's state dictionary
model.load_state_dict(torch.load('simple_nn_model_MNIST.pth'))

# Evaluate the model on the testing set
model.eval()
true_labels = []
predicted_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())


# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels) * 100
precision = precision_score(true_labels, predicted_labels, average='weighted') * 100
recall = recall_score(true_labels, predicted_labels, average='weighted') * 100
f1 = f1_score(true_labels, predicted_labels, average='weighted') * 100
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Print the evaluation metrics
print(f"Accuracy on the test set: {accuracy:.2f}%")
print(f"Precision on the test set: {precision:.2f}%")
print(f"Recall on the test set: {recall:.2f}%")
print(f"F1-score on the test set: {f1:.2f}%")

# print("Confusion Matrix:")
# print(conf_matrix)



# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()