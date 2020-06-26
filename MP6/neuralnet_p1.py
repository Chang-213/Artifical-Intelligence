# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        The network should have the following architecture (in terms of hidden units):
        in_size -> 128 ->  out_size
        """
        super(NeuralNet, self).__init__()
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_unit = nn.Linear(self.in_size, 128, True)
        self.out = nn.Linear(128, self.out_size, True)
        self.optimize = optim.SGD(self.get_parameters(), lr=self.lrate)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()


    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        # return self.net.parameters()
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        y = self.sig(self.out(self.relu(self.hidden_unit(x))))
        return y

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """

        #output = self(x)
        L = self.loss_fn(self(x), y)
        L.backward()
        self.optimize.step()
        self.optimize.zero_grad()
        return L

def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, 784) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M, 784) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """
    #initialize variables
    losses = []
    lrate = 1
    insize = len(train_set[0])
    outsize = 3
    net = NeuralNet(lrate, nn.CrossEntropyLoss(), insize, outsize)

    #standardize data
    sample_mean = train_set.mean()
    std = train_set.std()
    train_set = (train_set - sample_mean) / std
    dev_set = (dev_set - sample_mean) / std

    #train
    for x in range(n_iter):
        g_step = net.step(train_set, train_labels).item()
        losses.append(g_step)

    #set outputs
    dev_labels = net(dev_set)
    yhat_labels = dev_labels.argmax(1).detach()
    yhats = np.array(yhat_labels)
    #print(losses)
    return losses,yhats,net
