# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 18:36:57 2017

@author: Sreenivas
"""
import torch
import torchvision
from torch.autograd import Variable
from torchvision import models
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from sklearn.metrics import accuracy_score,confusion_matrix
import imageloading as Image
import numpy as np
import pandas as pd
#Training Settings
parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--data', metavar='DIR',
                    help='path to image directory')
parser.add_argument('--label',metavar='FILE',help='path to label file')
parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--testbatchsize', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--validbatchsize', type=int, default=64, metavar='N',
                    help='input batch size for validation (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
#I ahve made no cuda default True->change when gpu is set up
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

#Buiding convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,5,5)
        self.pool=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(5,5,5)
        self.fc1=nn.Linear(5*53*53,20)
        self.fc2=nn.Linear(20,25)
        self.fc3=nn.Linear(25,5)
        
        
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,5*53*53)
        x=F.relu(self.fc1(x))
        x=F.dropout(x,training=self.training)
        x=F.relu(self.fc2(x))
        x=F.dropout(x,training=self.training)
        x=self.fc3(x)
        return F.log_softmax(x)

#Train function
def train(model,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        if args.cuda:
            data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
#Test function
def test(model,test_loader):
    model.eval()
    test_loss=0
    correct=0
    for batch_idx,(data,target) in enumerate(test_loader):
        if args.cuda:
            data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)
        output=model(data)
        test_loss+=F.nll_loss(output,target,size_average=False).data[0]
        pred=output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
#Main function
if __name__=="__main__":
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    model=Net()
    if args.cuda:
        model.cuda()
    #optimizer and criterion for neural network    optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)
    #normalizing data
    normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform=transforms.Compose([transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    #taking directory with image folder from command line
    imagedir=args.data
    labelfile=args.label
    #calling class from imageloading
    imagedataset=Image.Imagedataset(imagedir,labelfile,transform)
    #splitting datasets into train,test and validation datasets
    inputfilelength=len(imagedataset)
    indices=list(range(inputfilelength))
    # splitting tensor dataset into 70% for training and 10% validation and 20% for testing
    split=int(np.floor(0.2*inputfilelength))
    trainvalid_idx,test_idx=indices[split:],indices[:split]
    trainvalid_sampler=torch.utils.data.sampler.SubsetRandomSampler(trainvalid_idx)
    test_sampler=torch.utils.data.sampler.SubsetRandomSampler(test_idx)
    trainvalidlength=len(trainvalid_sampler)
    trainvalidindices=list(range(trainvalidlength))
    trainvalidsplit=int(np.floor(0.1*trainvalidlength))
    train_idx,validation_idx=trainvalidindices[trainvalidsplit:],trainvalidindices[:trainvalidsplit]
    train_sampler=torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    validation_sampler=torch.utils.data.sampler.SubsetRandomSampler(validation_idx)
    #dataloader for train test and validation
    train_loader=torch.utils.data.DataLoader(imagedataset,sampler=train_sampler,batch_size=args.batchsize)
    test_loader=torch.utils.data.DataLoader(imagedataset,sampler=test_sampler,batch_size=args.testbatchsize)
    validation_loader=torch.utils.data.DataLoader(imagedataset,sampler=validation_sampler,batch_size=args.validbatchsize)
    #Training phase
    for epoch in range(0,args.epochs):
        train(model,train_loader,optimizer,epoch)
        test(model,test_loader)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

    


    
    
        
        