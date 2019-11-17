#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linear SVM with custom multiclass Hinge loss
"""

import os
import time
import argparse
import pickle
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,gradcheck
import torch.utils.data as utils

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from data_loader import get_data_loader
import matplotlib.pyplot as plt # for plotting

# Training
def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values
    Args:
        config: Configuration object containing the hyperparameters
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

###############################################################################
# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.
    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt("{}_train_accuracy.csv".format(path))
    val_err = np.loadtxt("{}_val_accuracy.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    train_f1 = np.loadtxt("{}_train_f1.csv".format(path))
    val_f1 = np.loadtxt("{}_val_f1.csv".format(path))

    plt.title("Train vs Validation Accuracy")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation F1 Score")
    plt.plot(range(1,n+1), train_f1, label="Train")
    plt.plot(range(1,n+1), val_f1, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend(loc='best')
    plt.show()


"""define network"""
class Net(nn.Module):
    def __init__(self,n_feature,n_class):
        super(Net, self).__init__()
        self.name = "SVM2"
        self.fc=nn.Linear(n_feature,n_class)
        torch.nn.init.kaiming_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias,0.1)
        
    def forward(self,x):
        flattened = x.view(-1, 40 * 1727)
        output=self.fc(flattened)
        return output
"""SVM loss
Weston and Watkins version multiclass hinge loss @ https://en.wikipedia.org/wiki/Hinge_loss
for each sample, given output (a vector of n_class values) and label y (an int \in [0,n_class-1])
loss = sum_i(max(0, (margin - output[y] + output[i]))^p) where i=0 to n_class-1 and i!=y
Note: hinge loss is not differentiable
      Let's denote hinge loss as h(x)=max(0,1-x). h'(x) does not exist when x=1, 
      because the left and right limits do not converge to the same number, i.e.,
      h'(1-delta)=-1 but h'(1+delta)=0.
      
      To overcome this obstacle, people proposed squared hinge loss h2(x)=max(0,1-x)^2. In this case,
      h2'(1-delta)=h2'(1+delta)=0
"""
class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        #print(output.requires_grad)
        #print(y.requires_grad)
        output_y=output[torch.arange(0,y.size()[0]).long(),y].view(-1,1)#view for transpose
        #margin - output[y] + output[i]
        loss=output-output_y+self.margin#contains i=y
        #remove i=y items
        loss[torch.arange(0,y.size()[0]).long(),y]=0
        #max(0,_)
        loss[loss<0]=0
        #^p
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        #add weight
        if(self.weight is not None):
            loss=loss*self.weight
        #sum up
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]#output.size()[0]
        return loss



def evaluate(model, loader, criterion):
    model.eval()

    training_loss=0
    training_f1=0
    corr = 0
    
    print("Validation")
    for batch_idx,(data,target) in enumerate(loader, 0):

        #print(data.shape, target.shape)
        #optimizer.zero_grad()
        output=model(data)
        tloss=criterion(output, target)
        training_loss+=tloss.item()
        #training_loss+=tloss[0]
        pred = output.max(1, keepdim=True)[1]

        target = target.cpu().numpy()
        pred = pred.cpu().numpy()

        f1 = f1_score(target, pred, average='macro') # labels
        corr += accuracy_score(target, pred, normalize=False)

        print(f1)
        training_f1+=f1
    print('Validation set avg loss: {:.4f}'.format(training_loss/len(loader)))
    print('Validation set avg micro-f1: {:.4f}'.format(training_f1/len(loader)))
    print(corr, len(loader.sampler))
    print('Validation set accuracy score: {:.4f}'.format(corr/len(loader.sampler)))

    return training_loss/len(loader), training_f1/len(loader), corr/len(loader.sampler)



def test_run():

    num_epochs, batch_sz, alpha = 11, 64, 1e-3

    np.random.seed(360)
    trn_loader, val_loader, test_loader = get_data_loader(batch_sz)
    #trn_loader,tst_loader,n_feature,n_class,y_train_list,y_test_list=syntheticData()
    print(len(trn_loader), len(trn_loader.sampler))
    model=Net(1727 * 40, 6)

    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    criterion=multiClassHingeLoss()
    #testloss=multiClassHingeLoss()

    """begin to train"""
    best_epoch_idx=-1
    best_f1=0.
    history=list()

    train_f1 = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_f1 = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    train_accuracy = np.zeros(num_epochs)
    val_accuracy = np.zeros(num_epochs)

    '''
    reload_epoch = 11
    train_f1[0: reload_epoch] = [0.3155, 0.5609, 0.6393, 0.6642, 0.6687, 
        0.5951, 0.7456, 0.7642, 0.6269, 0.7543, 0.7672]
    train_loss[0: reload_epoch] = [166.3827, 19.0027, 11.1734, 8.7483, 11.9797, 
        24.8095, 7.6915, 6.7495, 25.5270, 10.3461, 7.6481]
    train_accuracy[0: reload_epoch] = [0.8049, 0.8842, 0.9160, 0.9198, 0.9160, 
        0.8950, 0.9385, 0.9385, 0.9070, 0.9425, 0.9460]
    val_f1[0: reload_epoch] = [0.4001, 0.4063, 0.4212, 0.4040, 0.3558, 
        0.4576, 0.4838, 0.4189, 0.3566, 0.4744, 0.4578]
    val_loss[0: reload_epoch] = [91.7369, 60.3723, 70.5087, 65.1146, 93.4909, 
        88.0412, 68.0768, 81.8570, 99.3559, 98.9875, 103.9215]
    val_accuracy[0: reload_epoch] = [0.8146, 0.8628, 0.8489, 0.8569, 0.8066, 
        0.8664, 0.8759, 0.8569, 0.8350, 0.8839, 0.8737]
    

    model_path = get_model_name(model.name, batch_sz, alpha, reload_epoch-1)
    model.load_state_dict(torch.load(model_path))
    '''

    for epoch in range(0, num_epochs):

        model.train()
        training_loss=0
        training_f1=0

        corr = 0
        for batch_idx,(data,target) in enumerate(trn_loader, 0):

            #print(data.shape, target.shape)
            optimizer.zero_grad()
            output=model(data)
            tloss=criterion(output, target)
            training_loss+=tloss.item()
            #training_loss+=tloss[0]
            tloss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]

            target = target.cpu().numpy()
            pred = pred.cpu().numpy()

            f1 = f1_score(target, pred, average='macro') # labels

            #print(target, pred)

            acc=accuracy_score(target, pred, normalize=False)
            corr += acc
            print(f1, acc)
            training_f1+=f1
        print('Epoch: {}'.format(epoch))
        print('Training set avg loss: {:.4f}'.format(training_loss/len(trn_loader)))
        print('Training set avg micro-f1: {:.4f}'.format(training_f1/len(trn_loader)))
        print(corr, len(trn_loader.sampler))
        print('Training set accuracy score: {:.4f}'.format(corr/len(trn_loader.sampler)))
  
        train_f1[epoch] = training_f1/len(trn_loader)
        train_loss[epoch] = training_loss/len(trn_loader)
        train_accuracy[epoch] = corr/len(trn_loader.sampler)


        #conf_mat, precision, recall, f1=test(model, val_loader) 
        val_loss[epoch], val_f1[epoch], val_accuracy[epoch]=evaluate(model, val_loader, criterion) 
        #print(precision)
        # print('Validation set avg loss: {:.4f}'.format(val_loss))
        # print('Validation set avg micro-f1: {:.4f}'.format(f1))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(model.name, batch_sz, alpha, epoch)
        torch.save(model.state_dict(), model_path)


        #history.append((conf_mat, precision, recall, f1))
        if f1>best_f1:#save best model
            best_f1=f1
            best_epoch_idx=epoch
            torch.save(model.state_dict(),'best.model')


    #print('Best epoch:{}\n'.format(best_epoch_idx))
    #conf_mat, precision, recall, f1=history[best_epoch_idx]
    #print('conf_mat:\n',conf_mat)
    #print('Precison:{:.4f}\nRecall:{:.4f}\nf1:{:.4f}\n'.format(precision,recall,f1))    
        # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs+1)
    np.savetxt("{}_train_f1.csv".format(model_path), train_f1)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_train_accuracy.csv".format(model_path), train_accuracy)
    np.savetxt("{}_val_f1.csv".format(model_path), val_f1)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
    np.savetxt("{}_val_accuracy.csv".format(model_path), val_accuracy)


train = False
#test_run()

if train:
    test_run()
else:

    num_epochs, batch_sz, alpha = 11, 64, 1e-3
    path = get_model_name("SVM2", batch_sz, alpha, num_epochs-1)
    plot_training_curve(path)

    