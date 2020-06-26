# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    predicted_labels = []

    #set up dict with corresponding words
    positive_words = 0
    negative_words = 0
    positive_list = Counter()
    negative_list = Counter()
    counter = 0
    for label in train_labels:
        if(label == 1):
            positive_words = positive_words + len(train_set[counter])
            for word in train_set[counter]:
                positive_list[word] += 1
            counter += 1
        else:
            negative_words = negative_words + len(train_set[counter])
            for word in train_set[counter]:
                negative_list[word] += 1
            counter += 1

    #calculate P(Word|Type=Positive) and P(Word|Type=Negative)
    positive_probability = {}
    negative_probability = {}
    for word in positive_list:
        positive_probability[word] = numpy.log((positive_list[word] + smoothing_parameter) / (positive_words + smoothing_parameter*len(positive_list)))
    for word in negative_list:
        negative_probability[word] = numpy.log((negative_list[word] + smoothing_parameter) / (negative_words + smoothing_parameter*len(negative_list)))

    #calculate the labels of the Dev set
    pos_rev_list = []
    neg_rev_list = []
    for review in dev_set:
        pos_review = 0
        for word in review:
            if(word in positive_probability):
                pos_review = pos_review + positive_probability[word]
            else:
                pos_review = pos_review + numpy.log((0 + smoothing_parameter) / (positive_words + smoothing_parameter*len(positive_list)))
        pos_review = pos_review + numpy.log(pos_prior)
        pos_rev_list.append(pos_review)

    for review in dev_set:
        neg_review = 0
        for word in review:
            if(word in negative_probability):
                neg_review = neg_review + negative_probability[word]
            else:
                neg_review = neg_review + numpy.log((0 + smoothing_parameter) / (negative_words + smoothing_parameter*len(negative_list)))
        neg_review = neg_review + numpy.log((1-pos_prior))
        neg_rev_list.append(neg_review)

    #comparison to assign values
    temp_counter = 0
    for review in dev_set:
        if (pos_rev_list[temp_counter] > neg_rev_list[temp_counter]):
            predicted_labels.append(1)
            temp_counter += 1
        else:
            predicted_labels.append(0)
            temp_counter += 1

    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return predicted_labels
