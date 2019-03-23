
# -*- coding: utf-8 -*-
"""
Created on Thur Mar 21 08:20:07 2019

@author: marc
"""
#----------------------Imports------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision import *
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import time as t
import torch.optim as optim
from PIL import Image, ImageOps

from resnet import *


import torchvision.models as models
resnet18 = resnet18(pretrained=True)


#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = r'trainData'
    #val_path = r'valData'
    #test_path = r'testData'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    #valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    #val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

    #testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    #test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_data_loader #, val_data_loader, #test_data_loader



#--------------------Base Model----------------------------------------------------

class BaseModel(nn.Module):
    def __init__(self, input_size):
        super(BaseModel, self).__init__()
        self.name = "Base"
        self.input_size = ((input_size - 2)/2)
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.conv2 = nn.Conv2d(5, 7, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(int(7 * 147 * 147), 1000)
        self.fc2 = nn.Linear(1000,2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,int(7*147 * 147) )
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x



class ResNet(nn.Module):
    def __init__(self,):
        super(ResNet, self).__init__()
        self.name = "ResNet"
        self.fc1 = nn.Linear( 2768896,1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = x.view(-1, 2768896)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------



def get_accuracy(model,set_):
    label_ = [0]*(150)
    label_.extend([1]*(150))


    label = torch.tensor(label_).cuda()

    model = model.cuda()
    trainSet_,valSet_,__ = get_data_loader(150)
    if set_ == "train":
        data_ = trainSet_
    #elif set_ == "val":
        #data_ = valSet_


    correct = 0
    total = 0
    for img, _ in data_:
        b = torch.split(img,600,dim=3)
        img = torch.cat(b, 0).cuda()

        res = resnet18(img)
        output = model(res)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item() #compute how many predictions were correct
        total += img.shape[0] #get the total ammount of predictions
        break

    return correct / total



def train(mdl,epochs= 20,batch_size = 32,learning_rate =0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9)
    trainSet = get_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations

    label_ = [0]*(batch_size)
    label_.extend([1]*(batch_size))


    label = torch.tensor(label_).cuda()
    mdl = mdl.cuda()
    print("--------------Starting--------------")



    for epoch in range(epochs):  # loop over the dataset multiple times



        t1 = t.time()

        itera = 0
        for img,_ in iter(trainSet):


            b = torch.split(img,600,dim=3)


            img = torch.cat(b, 0).cuda()

            print(img.size())


            itera += batch_size*2

            res = resnet18(img)
            print(res.size())
            out = mdl(res)


            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            print(itera)
        break
        # Calculate the statistics
        train_acc.append(get_accuracy(mdl,"train"))

        #val_acc.append(get_accuracy(mdl,"val"))  # compute validation accuracy
        n += 1


        print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1])#, "And Validation Accuracy:",val_acc[-1])


        # Save the current model (checkpoint) to a file
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
        torch.save(mdl.state_dict(), model_path)

    iterations = list(range(1,epochs + 1))

    print("--------------Finished--------------")

    return iterations,train_acc, val_acc



def plot(iterations,train_acc, val_acc):
    plt.title("Training Curve")
    plt.plot(iterations, train_acc, label="Train")
    #plt.plot(iterations, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    #print("Final Validation Accuracy: {}".format(val_acc[-1]))
