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
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as numpy
import math
from collections import Counter





def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """



    # TODO: Write your code here
    predicted_labels = []

    #set up dict with corresponding words for unigram
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
        positive_probability[word] = numpy.log((positive_list[word] + unigram_smoothing_parameter) / (positive_words + unigram_smoothing_parameter*len(positive_list)))
    for word in negative_list:
        negative_probability[word] = numpy.log((negative_list[word] + unigram_smoothing_parameter) / (negative_words + unigram_smoothing_parameter*len(negative_list)))

    #calculate the labels of the Dev set
    pos_rev_list = []
    neg_rev_list = []
    for review in dev_set:
        pos_review = 0
        for word in review:
            if(word in positive_probability):
                pos_review = pos_review + positive_probability[word]
            else:
                pos_review = pos_review + numpy.log((0 + unigram_smoothing_parameter) / (positive_words + unigram_smoothing_parameter*len(positive_list)))
        pos_review = pos_review + numpy.log(pos_prior)
        pos_rev_list.append(pos_review)

    for review in dev_set:
        neg_review = 0
        for word in review:
            if(word in negative_probability):
                neg_review = neg_review + negative_probability[word]
            else:
                neg_review = neg_review + numpy.log((0 + unigram_smoothing_parameter) / (negative_words + unigram_smoothing_parameter*len(negative_list)))
        neg_review = neg_review + numpy.log((1-pos_prior))
        neg_rev_list.append(neg_review)

    #comparison to assign values
    # temp_counter = 0
    # for review in dev_set:
    #     if (pos_rev_list[temp_counter] > neg_rev_list[temp_counter]):
    #         predicted_labels.append(1)
    #         temp_counter += 1
    #     else:
    #         predicted_labels.append(0)
    #         temp_counter += 1

    #bigram dictionary
    positive_words_bigram = 0
    negative_words_bigram = 0
    positive_list_bigram = Counter()
    negative_list_bigram = Counter()
    counter_bigram = 0
    for label in train_labels:
        if(label == 1):
            positive_words_bigram = positive_words_bigram + (len(train_set[counter_bigram])-1)
            word_list = [train_set[counter_bigram][s:s+2] for s in range(len(train_set[counter_bigram])-1)]
            for word in word_list:
                positive_list_bigram[str(word)] += 1
            counter_bigram += 1
        else:
            negative_words_bigram = negative_words_bigram + (len(train_set[counter_bigram])-1)
            word_list = [train_set[counter_bigram][s:s+2] for s in range(len(train_set[counter_bigram])-1)]
            for word in word_list:
                negative_list_bigram[str(word)] += 1
            counter_bigram += 1

    #calculate P(Word|Type=Positive) and P(Word|Type=Negative) for bigram
    positive_probability_bigram = {}
    negative_probability_bigram = {}
    #print(positive_list_bigram)
    for word in positive_list_bigram:
        positive_probability_bigram[word] = numpy.log((positive_list_bigram[word] + bigram_smoothing_parameter) / (positive_words_bigram + bigram_smoothing_parameter*len(positive_list_bigram)))
    for word in negative_list_bigram:
        negative_probability_bigram[word] = numpy.log((negative_list_bigram[word] + bigram_smoothing_parameter) / (negative_words_bigram + bigram_smoothing_parameter*len(negative_list_bigram)))

    #calculate the labels of the Dev set
    pos_rev_list_bigram = []
    neg_rev_list_bigram = []
    for review in dev_set:
        pos_review = 0
        word_list = [review[s:s+2] for s in range(len(review)-1)]
        for word in word_list:
            if(str(word) in positive_probability_bigram):
                pos_review = pos_review + positive_probability_bigram[str(word)]
            else:
                pos_review = pos_review + numpy.log((0 + bigram_smoothing_parameter) / (positive_words_bigram + bigram_smoothing_parameter*len(positive_list_bigram)))
        pos_review = pos_review + numpy.log(pos_prior)
        pos_rev_list_bigram.append(pos_review)

    for review in dev_set:
        neg_review = 0
        word_list = [review[s:s+2] for s in range(len(review)-1)]
        for word in word_list:
            if(str(word) in negative_probability_bigram):
                neg_review = neg_review + negative_probability_bigram[str(word)]
            else:
                neg_review = neg_review + numpy.log((0 + bigram_smoothing_parameter) / (negative_words_bigram + bigram_smoothing_parameter*len(negative_list_bigram)))
        neg_review = neg_review + numpy.log((1-pos_prior))
        neg_rev_list_bigram.append(neg_review)

    #mix values
    pos_rev_list_mix = []
    neg_rev_list_mix = []
    mix_counter = 0
    for review in dev_set:
        pos_rev_list_mix.append((bigram_lambda * pos_rev_list_bigram[mix_counter]) + ((1-bigram_lambda) * pos_rev_list[mix_counter]))
        neg_rev_list_mix.append((bigram_lambda * neg_rev_list_bigram[mix_counter]) + ((1-bigram_lambda) * neg_rev_list[mix_counter]))
        mix_counter += 1

    #comparison to assign values
    temp_counter = 0
    for review in dev_set:
        if (pos_rev_list_mix[temp_counter] > neg_rev_list_mix[temp_counter]):
            predicted_labels.append(1)
            temp_counter += 1
        else:
            predicted_labels.append(0)
            temp_counter += 1
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return predicted_labels
