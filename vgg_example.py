import torch
import torch.nn as nn
import torch.nn.functional as F
import fused_add_relu_ext_openmp_simd as ext  # Import the custom C++ extension
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleVGG(nn.Module):
    def __init__(self, in_channels=3, input_size=32, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, input_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_size, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # example: call custom op between conv layers if you want
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleVGG(in_channels=1, input_size=28)  # For MNIST, use 1 input channel and 28x28 size

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Mean and std for MNIST
])

# Load MNIST datasets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

def training_loop():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad() # Clear gradients
            output = model(data)
            loss = criterion(output, target)
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        # Evaluation after each epoch
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad(): # Disable gradient calculation for evaluation
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%')


# usage
if __name__ == "__main__":
    inp = torch.randn(8, 1, 28, 28)  # batch 8
    out = model(inp)
    print(out.shape)

    # Own module test
    a = torch.randn(10)
    b = torch.randn(10)
    out = ext.fused_add_relu(a, b)
    print(out)

    training_loop()