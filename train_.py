import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import cv2
import glob
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


classes = ["red blood cell", "difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"]

labels = glob.glob('/home/ubuntu/train/*.txt')
print('Reading dataset started')
all_images = []
all_labels = []
for numb in range(len(labels)):
    with open(labels[numb], 'r') as f:
        data = f.read()
    labels_encoding = []
    for i in classes:
        if i in data.split('\n'):
            labels_encoding.append(1)
        else:
            labels_encoding.append(0)
    all_labels.append(labels_encoding)

    img = cv2.imread(labels[numb].replace('.txt', '.png'))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Turn into greyscale
        img = cv2.resize(img, (512, 512))  # Resize according to cnn archetecture
    except:
        print("An exception occurred", labels[numb])
    all_images.append(img)

    print(numb, labels[numb])


all_labels = np.array(all_labels)
all_images = np.array(all_images)

all_images = all_images.reshape(all_images.shape[0], 1, all_images.shape[1],
                                all_images.shape[1])  # converting(n,512,512)>(n,1,512,512)
print(all_images.shape)

from sklearn.model_selection import train_test_split

train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2)

import torch
from torch.utils.data import Dataset, DataLoader


# soruce:https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()  # converting torch tensors from numpy array
        self.target = torch.from_numpy(target).float()
        self.transform = transform

    def __getitem__(self, index):  # making a function to get data by index batch number
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


dataset_train = MyDataset(train_images, train_labels)
dataset_test = MyDataset(test_images, test_labels)

#load data
loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, pin_memory=torch.cuda.is_available())
loader_test = DataLoader(dataset_test, batch_size=8, shuffle=True, pin_memory=torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# source:https://github.com/vatsalsaglani/ApparelClassifier/blob/master/train.py
class Net(nn.Module):
    def __init__(self, classes_number):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.convnorm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.convnorm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((2, 2))
        self.linear1 = nn.Linear(30 * 30 * 64, 1500)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(150, classes_number)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = F.relu(self.linear1(x.view(x.size(0), -1)))
        x = self.drop(x)
        x = self.linear2(x)
        out = torch.sigmoid(x)
        return out

model = Net(len(classes))

criterion = nn.BCELoss()  # Binary cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # optimization function
# optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)



model.to(device)
print(model)


import time

# #Train the network
for epoch in range(55):  # loop over the dataset
    time.sleep(10)
    running_loss = 0.0
    for i, data in enumerate(loader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i / (len(loader_train) - 1) == 1:
            print('Epoch:', epoch, ' loss:', running_loss)

    torch.save(model.state_dict(), "model_mbucalossi50.pt")

print('Finished Training')


Right_predicted = 0
for i, data in enumerate(loader_test, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(inputs)

    outputs[outputs > 0.49] = 1
    outputs[outputs < 0.50] = 0
    Right_predicted = Right_predicted + len(torch.nonzero(outputs == labels))

Accuracy = (Right_predicted / (len(test_labels) * 7)) * 100
print('Accuracy:', Accuracy)

