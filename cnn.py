import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# Load MNIST Dataset
train_data = datasets.MNIST('./Data', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]), download=True)
test_data = datasets.MNIST('./Data', train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]), download=True)
print("\n",train_data)
print("\n",test_data)

# Data Train Parameters
batch_size = 128
epochs = 25
learning_rate = 0.001

# Load Data into DataLoader
train_dataLoader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataLoader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check device config
device = ('cuda' if torch.cuda.is_available() else 'CPU')
print("Utilizing", device)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(stride=2, kernel_size=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(stride=2, kernel_size=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            nn.Tanh())

        self.fc1 = nn.Linear(in_features=120, out_features=84)

        self.fc2 = nn.Linear(in_features=84, out_features=10)

        self.act1 = nn.Tanh()
        self.avgpooling = nn.AvgPool2d(stride=2, kernel_size=2)
        self.softmax = nn.Softmax()

    def forward(self, out):
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.fc2(out)

        return out


# Initialize Model
model = Model().to(device)
print("\n",model)


# Set LOSS and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Total Step
total_step = len(train_dataLoader)
epoch_loss = []

# Training Loop
for epoch in range(epochs):

    model.train()
    total_loss = 0
    i = 0
    with tqdm(train_dataLoader, unit="batch") as tepoch:

        # for i, (images, labels) in enumerate(train_dataLoader):
        for images, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            image = images.to(device)
            label = labels.to(device)
            i+=1
            # Forward PASS
            output = model(image)
            predictions = output.argmax(dim=1, keepdim=True).squeeze()
            loss = loss_fn(output, label)
            # total_loss += loss

            # Backpropagation (Weight UPDATE)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = (predictions == label).sum().item()
            accuracy = correct / batch_size

            tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
        # epoch_loss = epoch_loss.append(total_loss/i)

torch.save(model.state_dict(), '../ML-Transfer_Learning/Model')

# Plot Training loss
# plt.plot(epochs, epoch_loss)


def accuracy(model):
    # Test the Model
    model.eval()
    num_samples = num_correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataLoader):
            test_image = images.to(device)
            test_label = labels.to(device)

            outputs = model(test_image)
            _, prediction = outputs.max(1)
            num_correct += (prediction == test_label).sum()
            num_samples += prediction.size(0)
            acc = (num_correct.item() / num_samples) * 100

        print("Model has an accuracy of {}%".format(acc))


accuracy(model=model)
# plt.show()
