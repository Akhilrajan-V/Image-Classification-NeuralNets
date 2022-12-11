#!/usr/bin/env python
"""
@author: Akhilrajan V
"""
import PIL
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


train_data_path = './Data_TransferLearning/training'
test_data_path = './Data_TransferLearning/validation'

# TRAINING PARAMETERS
batch_size = 32
epochs = 10
learning_rate = 0.001

# Check device config
device = ('cuda' if torch.cuda.is_available() else 'CPU')

imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])


def normalize(im):
    #im = im.astype(np.float32)/255.
    im = im/255.
    """Normalizes images with Imagenet stats."""
    return (im - imagenet_stats[0])/imagenet_stats[1]


def load_data(image_folder='./Data_TransferLearning/training'):
    images = []
    labels = []
    for class_id, folder in enumerate(sorted(os.listdir(image_folder))):
        print("Accessing Folder {}".format(folder))
        for file in sorted(os.listdir(image_folder + '/' + folder)):
            img = np.array(PIL.Image.open(image_folder + '/' + folder + '/'+file).resize((256, 256)))
            # print(img.shape)
            img = normalize(img)
            img = np.moveaxis(img, 2, 0)
            images.append(img)
            labels.append(class_id)
    images = np.array(images)
    np.save('val_data', images)
    labels = np.array(labels)
    np.save('val_label', labels)
    return images, labels


# train_img, train_labels = load_data(train_data_path)

train_img = np.load('data.npy')
train_labels = np.load('label.npy')

# test_img, test_labels = load_data(test_data_path)

test_img = np.load('val_data.npy')
test_labels = np.load('val_label.npy')


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)


# Train Data Generator
x_train_tensor = torch.from_numpy(train_img).float()
y_train_tensor = torch.from_numpy(train_labels).float()
print(x_train_tensor.shape)
# print(y_train_tensor.shape)
# train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_data = CustomDataset(x_train_tensor, y_train_tensor)

# Test Data Generator
x_test_tensor = torch.from_numpy(test_img).float()
y_test_tensor = torch.from_numpy(test_labels).float()
# test_data = TensorDataset(x_test_tensor, y_test_tensor)
test_data = CustomDataset(x_train_tensor, y_train_tensor)
# print(train_data)

# DATA LOADERS
train_dataLoader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataLoader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


# Custom Convolution Neural Net
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0)
        self.fc = nn.Linear(in_features=128*13*13, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x.float()


# Initialize Custom CNN Model
model = Model().to(device)
print("\n",model)

"""
    ------- TRANSFER LEARNING  MODEL -------
"""
# Create Transfer Learning model based on VGG16 Pretrained Architecture
class TransferLearningModel:
    def __init__(self):
        super(TransferLearningModel, self).__init__()
        self.weights = models.VGG16_Weights.DEFAULT
        self.backbone = models.vgg16(weights=self.weights)
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_features = self.backbone.classifier[6].in_features
        features = list(self.backbone.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, 10)])  # Add our layer with 4 outputs
        self.backbone.classifier = nn.Sequential(*features)  # Replace the model classifier


# Initialize Custom CNN Model
transferLearn = TransferLearningModel().backbone.to(device)
print("\n",transferLearn)

# Create LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()


# Set OPTIMIZER to optimize model
def optim_initializer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer


# Total Step
total_step = len(train_dataLoader)
epoch_loss = []


# TRAIN MODULE
def train(model):
    optimizer = optim_initializer(model)

    # Training Loop
    for epoch in range(epochs):

        model.train()
        total_loss = 0
        i = 0
        with tqdm(train_dataLoader, unit="batch") as tepoch:

            # for i, (images, labels) in enumerate(train_dataLoader):
            # Per EPOCH Train Loop (All mini-batches are used for training which completes one epoch)
            for images, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                image = images.to(device)
                label = labels.to(device)
                i += 1

                # Forward PASS
                output = model(image)
                predictions = output.argmax(dim=1, keepdim=True).squeeze()
                loss = loss_fn(output, label.long())
                # total_loss += loss

                # BACK PROPAGATION (Weight UPDATE)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                correct = (predictions == label).sum().item()
                accuracy = correct / batch_size

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
            # epoch_loss = epoch_loss.append(total_loss/i)


# Plot Training loss
# plt.plot(epochs, epoch_loss)


# TEST MODULE
def accuracy(model, tl=False):
    name = "CNN model"
    if tl:
        name = "Transfer Learning trained model"
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

        print("\nThe custom {} has an accuracy of {}%".format(name, acc))


if __name__ == "__main__":
    print("\nTraining Custom CNN model\n")
    train(model=model)
    accuracy(model=model)
    print("\nTraining using Transfer Learning CNN model\n")
    train(model=transferLearn)
    accuracy(transferLearn)

