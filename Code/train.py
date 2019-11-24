import torch
from torch import nn, optim

import numpy as np
import time
import matplotlib.pyplot as plt

from data_loader import get_data_loader

global label_to_class

def train_net(net, batch_size=64, learning_rate=0.01, num_epochs=30, reload_epoch=0, model_path=None):
    # Fixed PyTorch random seed for reproducible result
    torch.manual_seed(360)
    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader, val_loader, test_loader, class_to_label = get_data_loader(batch_size)

    global label_to_class 
    label_to_class = {v: k for k, v in class_to_label.items()}
    ########################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    ########################################################################
    # Set up some numpy arrays to store the training/test loss/erruracy
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    # Read prvious record if reload_epoch is not 0
    if model_path is not None:
        train_err[0:reload_epoch] = np.loadtxt("{}_train_err.csv".format(model_path))
        val_err[0:reload_epoch]= np.loadtxt("{}_val_err.csv".format(model_path))
        train_loss[0:reload_epoch] = np.loadtxt("{}_train_loss.csv".format(model_path))
        val_loss[0:reload_epoch] = np.loadtxt("{}_val_loss.csv".format(model_path))

    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()
    for epoch in range(reload_epoch, num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        net.train()
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            #findMisclassified(outputs, labels)
            loss = criterion(outputs, labels.long())
            print(loss)
            loss.backward()
            optimizer.step()
            # Calculate the statistics
            corr = outputs.max(dim=1).indices.long() != labels
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        net.eval()
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}\n\n").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

    # plot_training_curve(model_path)

###############################################################################
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

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        corr = outputs.max(dim=1).indices.long() != labels
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    findMisclassified(outputs, labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss

###############################################################################
# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))

    # err to acc
    train_acc = 1 - train_err
    val_acc = 1 - val_err

    plt.title("Train vs Validation Accuracy")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
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

###############################################################################
# Find Misclassified Points
def findMisclassified(out, label):
    classes, class_freq = np.unique(label, return_counts=True)
    err = {c: 0 for c in classes}

    # Find incorrect heuristic
    for i in range(len(label)):
        l = label[i].item()
        p = out[i].max(dim=0).indices.long().item()
        if l != p:
            err[l] += 1

    # From label to class
    err_class = {label_to_class[k] : v for k, v in err.items()}

    # Find accuracy rate for each class
    acc_rate = {label_to_class[classes[i]] : round(1 - err[classes[i]]/class_freq[i], 4) for i in range(len(class_freq))}

    print("Misclassify Info: {}".format(err_class))
    print("Class Accuracy Rate: {}".format(acc_rate))
