'''
Epoch: 1 Loss: 0.2433358688217898 Accuracy: 0.9287666666666666
Epoch: 2 Loss: 0.1113192329746671 Accuracy: 0.9662
Epoch: 3 Loss: 0.08071125643824538 Accuracy: 0.9750833333333333
Loss: 0.10431371338868142 Accuracy: 0.9704
'''

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load mnist dataset
# Convert pixel value from 0~255 to 0~1 (Normalization)
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Define the size
batch_size = 32

# Create PyTorch datasets
train_datasets = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

test_datasets = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

# Initializer and parameters
torch.manual_seed(1)

w1 = torch.nn.Parameter(torch.empty(784,392))
b1 = torch.nn.Parameter(torch.zeros(392))

w2 = torch.nn.Parameter(torch.empty(392,196))
b2 = torch.nn.Parameter(torch.zeros(196))

w3 = torch.nn.Parameter(torch.empty(196,98))
b3 = torch.nn.Parameter(torch.zeros(98))

w4 = torch.nn.Parameter(torch.empty(98,49))
b4 = torch.nn.Parameter(torch.zeros(49))

w5 = torch.nn.Parameter(torch.empty(49,10))
b5 = torch.nn.Parameter(torch.zeros(10))

# He Normal initialization
torch.nn.init.kaiming_normal_(w1,nonlinearity="relu")
torch.nn.init.kaiming_normal_(w2,nonlinearity="relu")
torch.nn.init.kaiming_normal_(w3,nonlinearity="relu")
torch.nn.init.kaiming_normal_(w4,nonlinearity="relu")
torch.nn.init.kaiming_normal_(w5,nonlinearity="relu")

# Forward Propagation
def forward(x):

    # Flatten images into a one dimension tensor
    x = x.reshape(x.shape[0],784)

    z1 = F.relu(
        torch.matmul(x,w1)+b1
    )

    z2 = F.relu(
        torch.matmul(z1,w2)+b2
    )

    z3 = F.relu(
        torch.matmul(z2,w3)+b3
    )

    z4 = F.relu(
        torch.matmul(z3,w4)+b4
    )

    y_pre = torch.matmul(z4,w5)+b5

    return y_pre

# Loss function
# CrossENtropyLoss
loss_fn = torch.nn.CrossEntropyLoss()

# Training variable
training_variables= [
    w1,w2,w3,w4,w5,
    b1,b2,b3,b4,b5
]

# Optimizer
optimizer = torch.optim.Adam(
    training_variables,
    lr = 0.001
)

# Training loop
epochs = 3

for epoch in range(epochs):
    total_loss = 0.0
    correct = 0
    total = 0

    for x_batch,y_batch in train_datasets:
        # Clear previous gradients
        optimizer.zero_grad()

        # Forward propagation
        y_pre = forward(x_batch)

        # Loss
        loss = loss_fn(
            y_pre,
            y_batch
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Calculate accuracy
        pred = torch.argmax(y_pre, dim=1)

        correct += torch.sum(
            (pred == y_batch).float()
        ).item()

        total += y_batch.size(0)

        total_loss += loss.item()

    accuracy = correct / total

    print(
        "Epoch:", epoch + 1,
        "Loss:", total_loss / len(train_datasets),
        "Accuracy:", accuracy
    )

# Test model
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():

    for x_batch, y_batch in test_datasets:

        y_pre = forward(x_batch)

        loss = loss_fn(
            y_pre,
            y_batch
        )

        pred = torch.argmax(y_pre, dim=1)

        correct += torch.sum(
            (pred == y_batch).float()
        ).item()

        total += y_batch.size(0)

        test_loss += loss.item()

test_accuracy = correct / total

print(
    "Loss:", test_loss / len(test_datasets),
    "Accuracy:", test_accuracy
)
