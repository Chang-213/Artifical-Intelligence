# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2018
import numpy as np
"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    #set variables
    correct_labels = np.zeros(len(train_labels))
    (m , n) = train_set.shape
    W = np.zeros(n)
    b = 0
    #have array where values are 1 or -1, not 1 or 0
    for label in range(len(train_labels)):
        if train_labels[label] == 0:
            correct_labels[label] = -1
        else:
            correct_labels[label] = 1
    #iterate thru and set up W and b
    for epoch in range(max_iter):
        for image in range(len(train_set)):
            feature = train_set[image]
            wtx = np.dot(W,feature)
            presgn = wtx + b
            y = np.sign(presgn)
            if(y != correct_labels[image]):
                W = W + (learning_rate*correct_labels[image]*feature)
                b = b + (learning_rate*correct_labels[image]*1)
    # return the trained weight and bias parameters
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    classify = []
    for i in range(len(dev_set)):
        feature = dev_set[i]
        wtx = np.dot(W,feature)
        presgn = wtx + b
        y = np.sign(presgn)
        if(y == 1):
            classify.append(1)
        else:
            classify.append(0)
    # Train perceptron model and return predicted labels of development set
    return classify

def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    s = 1 / (1 + np.exp(-x))
    return s

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    #set variables
    correct_labels = np.zeros(len(train_labels))
    (m , n) = train_set.shape
    W = np.zeros(n)
    b = 0
    #have array where values are 1 or -1, not 1 or 0
    for label in range(len(train_labels)):
        if train_labels[label] == 0:
            correct_labels[label] = -1
        else:
            correct_labels[label] = 1
    #iterate thru and set up W and b
    for epoch in range(max_iter):
        gradient = 0
        activation = 0
        labels = 0
        for image in range(len(train_set)):
            feature = train_set[image]
            wtx = np.dot(W,feature)
            presgn = wtx + b
            y = sigmoid(presgn)
            activation = activation + y
            g = (y - train_labels[image]) * feature
            gradient = gradient + g
            labels = labels + train_labels[image]
        W = W - (learning_rate*gradient)/len(train_set)
        b = b - (learning_rate * (activation-labels) )/len(train_set)
    # return the trained weight and bias parameters
    return W, b

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    W, b = trainLR(train_set, train_labels, learning_rate, max_iter)
    classify = np.zeros(len(dev_set))
    for i in range(len(dev_set)):
        feature = dev_set[i]
        wtx = np.dot(W,feature)
        presgn = wtx + b
        y = sigmoid(presgn)
        if(y >= 0.5):
            classify[i] = 1
        else:
            classify[i] = 0
    # Train LR model and return predicted labels of development set
    return classify

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    return []
