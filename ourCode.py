# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 21:30:29 2019

@author: Vincent
"""

from __future__ import print_function
import torch
import pandas as pd
import numpy as np
from scipy.stats import zscore
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import cv2 as cv
import matplotlib.pyplot as plt
from torchvision.models import vgg

#either uses GPU or CPU, depending if cuda is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#the neural networks we are testing
class Net1(nn.Module):
    
    def __init__(self, num_classes=10):
        super(Net1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(50*17*17, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = out.reshape(-1, 50*17*17)
        out = self.fc(out)
        return out

class Net2(nn.Module):
    def __init__(self, num_classes=10):
        super(Net2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))     
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))   
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        in_size=x.size(0)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = self.layer5(out)
        #print(out.shape)
        out = out.view(in_size, -1)
        out = self.fc(out)
        return out

class Net3(nn.Module):
    def __init__(self, num_classes=10):
        super(Net3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, padding=2, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, padding=2, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=2, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=2, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))     
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=2, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))  
        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=4, padding=2, stride=1),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.MaxPool2d(2,2))  
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        in_size=x.size(0)
        out = self.layer1(x)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = self.layer5(out)
        #print(out.shape)
        out = self.layer6(out)
        #print(out.shape)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
#A custom dataset for our images and labels
class CustomDataset(torch.utils.data.Dataset):
     def __init__(self, X_tensor, y_tensor):
         self.X_tensor = X_tensor
         self.y_tensor = y_tensor
         return
     def __getitem__(self, index):
         self.img = self.X_tensor[index]
         label = self.y_tensor[index]
         return (self.img,label)
     def __len__(self):
         return self.X_tensor.size()[0]

#function that preproceses our images
def preprocess(images):
    
    images = images/255
    images[images < 0.90] = 0
    images[images > 0] = 1
    
    for i in range(len(images)):
        
        image = images[i]        
        image=np.array(image, dtype='uint8')  
        
        contours,_ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        
        largest_area = 0
        for contour in contours:
            
            coor1, coor2, width, height = cv.boundingRect(contour)
            
            side = max(width, height)
            
            area = side * side
            
            if area > largest_area:
                largest_area = area
                larC1, larC2, larW, larH = coor1, coor2, width, height
                        
        
        largest_digit = image[larC2:larC2+larH,larC1:larC1+larW]
        
        [rows,cols] = largest_digit.shape
        if rows>cols:
            difference = rows - cols
            if difference%2 !=0:
                difference += 1 
            image = np.concatenate((np.zeros((rows,int(difference/2))),largest_digit,np.zeros((rows,int(difference/2)))), axis=1)
        elif rows<cols:
            difference = cols - rows
            if difference%2 !=0:
                difference += 1 
            image = np.concatenate((np.zeros((int(difference/2),cols)),largest_digit,np.zeros((int(difference/2),cols))), axis=0)
        else:
            image = largest_digit
            
        image = cv.resize(image,(64, 64), interpolation = cv.INTER_AREA)
        images[i] = image
    
    return images

#builds the dataloaders for our images and labels
def build_loaders():
        
    train_images = pd.read_pickle('train_images.pkl')
    train_labels = pd.read_csv('train_labels.csv')

    '''
    img_idx = 3
    plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
    plt.imshow(train_images[img_idx])
    '''

    train_images = preprocess(train_images)
    
    '''
    img_idx = 17
    plt.title('Label: {}'.format(train_labels.iloc[img_idx]['Category']))
    plt.imshow(train_images[img_idx])    
    '''
    
    train_images = train_images.reshape((40000,1,64,64))
    
    #Pre-Process the csv-files of the labels    
    train_labels = train_labels.drop(['Id'], axis=1)
    train_labels = train_labels.values
    train_labels = np.reshape(train_labels, (-1))

    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.10, random_state=2)
    #X_test, X_rem, y_test, y_rem = train_test_split(X_test, y_test, test_size=0.90, random_state=2)
    
    X_tensor_train = torch.tensor(X_train)
    y_tensor_train = torch.tensor(y_train)
    X_tensor_test = torch.tensor(X_test)
    y_tensor_test = torch.tensor(y_test)

    train_dataset = CustomDataset(X_tensor_train, y_tensor_train)
    test_dataset = CustomDataset(X_tensor_test, y_tensor_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=2, shuffle=True)
    
    return train_loader, test_loader, X_tensor_test, y_tensor_test

#function that predicts the label for every image in the test set. Returns a
#panda dataframe
def predictTest():
    
     test_images = pd.read_pickle('test_images.pkl')
     
     img_idx = 4
     plt.imshow(test_images[img_idx])
     
     test_images = preprocess(test_images)
     test_images = test_images.reshape(10000,1,64,64)
     test_images = torch.tensor(test_images)
     
     predicted = np.zeros((len(test_images)))
     for i in range(len(test_images)):
         image = test_images[i,:,:,:]
         image = image.reshape(1,1,64,64)
         output = net(image)
         _, predicted[i] = torch.max(output.data, 1)
     
     predicted = predicted.astype(int)
     dataframe = pd.DataFrame({'Category':predicted})
     dataframe.to_csv("submission.csv")
       
     return predicted
    
#loop that runs the adam algorithm
def makeItLearn(epoch, best_model_accuracy):
        
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay = 0)
    
    #for each batch, update the weights, using the Adam Algorithm    
    total_step = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(images)
        loss = cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        
        #prints the loss
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    #compute the accuracy of our model on the validation set
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))        
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                        
            for i in range(2):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    accuracy = 100 * correct / total
    print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
    
    #if the accuracy obtained with these new weights is better than whatever
    #we had as our best model before, we save the best_model in a .ckpt file
    if(accuracy) > best_model_accuracy:
        
        print("DAMN!!! Good Job <3!!!")
        torch.save(net.state_dict(), 'best_model.ckpt')
        best_model_accuracy = accuracy
    
    #gives the accuracy of our model on each of the labels, separately
    for i in range(10):
        print('Accuracy of %d : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))
    
    loss_accuracy_lr[epoch, 0] = loss.item()               #loss on training set
    loss_accuracy_lr[epoch, 1] = (100 * correct / total)   #accuracy on valid set
    loss_accuracy_lr[epoch, 2] = epoch                     #nb of epochs
    
    return best_model_accuracy

#return the prediction on our validation set, which an array of tensors
#as input (not the dataloader)
def getPredictionsValidation(test_images):
    predicted = np.zeros((len(test_images)))
    for i in range(len(test_images)):
        image = test_images[i,:,:,:]
        image = image.reshape(1,1,64,64)
        output = net(image)
        _, predicted[i] = torch.max(output.data, 1)
    
    return predicted

#return a list of the wrongly labeled images in the test set (aka validation set)
def getWrongs(predicted, y_tensor_test):
    
    y_tensor_test = y_tensor_test.numpy()
    
    wrongs = []
    for i in range(len(predicted)):
        if predicted[i] != y_tensor_test[i]:
            wrongs.append(i)
        
    return wrongs

#function that pre process the images for them to be used on the VGG model
def VGG_get_loaders():
    
    train_images = pd.read_pickle('train_images.pkl')
    train_labels = pd.read_csv('train_labels.csv')

    train_images = preprocess(train_images)
    train_images = train_images.reshape((40000,1,64,64))
    train_images_3 = np.zeros((40000,3,64,64))
    
    train_images_3[:,0,:,:] = train_images
    train_images_3[:,1,:,:] = train_images
    train_images_3[:,2,:,:] = train_images
    train_images = train_images_3
    
    #Pre-Process the csv-files of the labels    
    train_labels = train_labels.drop(['Id'], axis=1)
    train_labels = train_labels.values
    train_labels = np.reshape(train_labels, (-1))

    X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.10, random_state=2)
    #X_test, X_rem, y_test, y_rem = train_test_split(X_test, y_test, test_size=0.90, random_state=2)
        
    X_tensor_train = torch.tensor(X_train)    
    y_tensor_train = torch.tensor(y_train)
    X_tensor_test = torch.tensor(X_test)
    y_tensor_test = torch.tensor(y_test)

    train_dataset = CustomDataset(X_tensor_train, y_tensor_train)
    test_dataset = CustomDataset(X_tensor_test, y_tensor_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=2, shuffle=True)
    
    return train_loader, test_loader, X_tensor_test, y_tensor_test
              
'''
MAIN
'''
cross_entropy = nn.CrossEntropyLoss()

#LOAD MODEL
net = Net2()
#net = Net3()
#net = vgg.vgg16(pretrained=True)

#SAVES CURRENT BEST ACCURACY + LOAD BEST MODEL
#best_model_accuracy = 0
#net.load_state_dict(torch.load('90_5.ckpt'))

#GET INDICES OF WRONGLY PREDICTED IMAGES IN VALIDATION SET
#_,_, X_tensor_test, y_tensor_test = build_loaders()
#predicted = getPredictionsValidation(X_tensor_test)
#wrongs = getWrongs(predicted, y_tensor_test)

#GET PREDICTED LABELS FOR THE TEST SET
#predicted = predictTest() 

#TRAIN THE MODEL

#load the datasets into a loader
train_loader, test_loader,_,_ = build_loaders()  

num_epochs = 20
learning_rate = 0.0007
loss_accuracy_lr = np.zeros((20,3))   
#we compare the accuracy of our newly trained model to our previous best model
for epoch in range(num_epochs):
    best_model_accuracy = makeItLearn(epoch, best_model_accuracy)
