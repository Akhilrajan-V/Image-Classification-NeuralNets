#!/usr/bin/env python

"""
@author: Akhilrajan V
"""

import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
import numpy as np


# Load MNIST Dataset
train_data = datasets.MNIST('ML-Transfer_Learning/Data', train=True, download=True)
test_data = datasets.MNIST(root='../Data', train=False, download=True)
print(train_data)
print("\n",test_data)

t = np.array(train_data.data)
tst = np.array(test_data.data)
train = []
test = []
for i in range(t.shape[0]):
    train.append(t[i].flatten())
train = np.array(train)

for j in range(tst.shape[0]):
    test.append(tst[j].flatten())
test = np.array(test)

# Check GPU availability
if torch.cuda.is_available():
    print("Running on :", "cuda")
    X = torch.from_numpy(train)
    X_test = torch.from_numpy(test)

else:
    X = train
    X_test = test

X_linear = preprocessing.normalize(X)

print("Data Type :", X.dtype)

Y = train_data.targets
Y_test = test_data.targets
# print("Y :", Y.shape)


def choose_kernel(c):
    if c == 1:
        kernel = "linear"
        print("\nTraining Classifier...")
        linear_clf = svm.LinearSVC()
        linear_clf.fit(X, Y)
        print("\nModel trained")
        Y_pred = linear_clf.predict(X_test)
        log(Y_pred, kernel)

    if c == 2:
        kernel = "polynomial"
        print("\nTraining Classifier...")
        poly_clf = svm.SVC(kernel='poly')
        poly_clf.fit(X, Y)
        print("Model trained")
        Y_pred = poly_clf.predict(X_test)
        log(Y_pred, kernel)

    if c == 3:
        kernel = "RBF"
        print("\nTraining Classifier...")
        rbf_clf = svm.SVC(kernel='rbf')
        rbf_clf.fit(X, Y)
        print("Model trained")
        Y_pred = rbf_clf.predict(X_test)
        log(Y_pred, kernel)


def accuracy(y, y_pred):
    correct = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            correct += 1
    acc = (correct / len(y_pred)) * 100
    return acc


def log(Y_pred, k):
    print("The predicted Data is :")
    print(Y_pred)
    print("The actual data is:")
    print(np.array(Y_test))
    print(f"The {k} Kernel SVM Classifier is {accuracy(Y_test, Y_pred)}% accurate")


if __name__ == "__main__":
    print("\nChoose SVM Kernel")
    print("\n 1.Linear \t 2.Polynomial \t 3.RBF")
    choice = int(input("\nEnter the number corresponding to the classifier you want to use \nChoose Classifier: "))
    if choice == 1 or choice == 2 or choice == 3:
        choose_kernel(choice)
    else:
        print("Wrong Input...Run Again")
        exit()
