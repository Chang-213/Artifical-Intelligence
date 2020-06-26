"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

from collections import Counter
from collections import defaultdict
import numpy as np

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    predicts = []

    #set up dict
    tag_counter = Counter()
    word_dict = defaultdict(list)
    for sentence in train:
        for word, tag in sentence:
            tag_counter[tag] += 1
            word_dict[word].append(tag)

    #set up most tags per word
    word_tag_assign = {}
    for word in word_dict:
        word_tag_counter = Counter()
        for tag in word_dict[word]:
            word_tag_counter[tag] +=1
        word_tag_assign[word] = word_tag_counter.most_common(1)[0][0]

    #assign test tag values
    for sentence in test:
        sent = []
        for word in sentence:
            if word in word_tag_assign:
                sent.append((word, word_tag_assign[word]))
            else:
                sent.append((word, tag_counter.most_common(1)[0][0]))
        predicts.append(sent)

    #raise Exception("You must implement me")
    return predicts


def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''

    predicts = []
    smooth = 0.00001

    #set up dict
    tag_counter = Counter()
    vocab_size = Counter()
    starting_tag_counter = Counter()
    word_dict = defaultdict(list)
    tag_dict = defaultdict(list)
    for sentence in train:
        starting_tag_counter[sentence[0][1]] += 1
        for word, tag in sentence:
            tag_counter[tag] += 1
            vocab_size[word] += 1
            word_dict[word].append(tag)
            tag_dict[tag].append(word)
    #this stores count(tag) : tag_counter[tag]
    #this stores vocab_size: len(vocab_size)
    #this stores no_of_tags: len(tag_counter)
    #this stores no_of_sentences: len(train)
    #this stores count(tag_i,starting_position) : starting_tag_counter

    #emission probability helper
    count_word_tag = {}
    for tag in tag_dict:
        word_tag_counter = Counter()
        for word in tag_dict[tag]:
            word_tag_counter[word] +=1
        #this stores Count(word,tag)
        count_word_tag[tag] = word_tag_counter
    #this is how you get the count(word,tag)
    #count_word_tag['X']['a']

    #transition probability helper
    count_prev_current = Counter()
    for sentence in train:
        prev_tag = 'NULL'
        for word, tag in sentence:
            count_prev_current[(prev_tag, tag)] += 1
            prev_tag = tag

    #test assignments
    initial_prob = defaultdict(list)
    trans_prob = defaultdict(list)
    for tag in tag_counter:
        initial_prob[tag].append(((starting_tag_counter[tag]+smooth)/(len(train)+(smooth*abs(len(tag_counter))))))
        for prev_tag in tag_counter:
            trans_prob[(prev_tag, tag)].append(((count_prev_current[(prev_tag, tag)] + smooth)/(tag_counter[prev_tag]+(smooth*abs(len(tag_counter))))))

    for sentence in test:
        emission_prob = defaultdict(list)
        word_val = defaultdict(list)
        first_word = 0
        trellis = defaultdict(list)
        last_word = 'NONE'
        for x in range(len(sentence)):
            temp = defaultdict(list)
            for tag in tag_counter:
                emission_prob[(sentence[x], tag)].append(((count_word_tag[tag][sentence[x]] + smooth)/(tag_counter[tag] + (smooth * abs(len(vocab_size)+1)))))
                if first_word == 0:
                    value = numpy.log(initial_prob[tag]) + numpy.log(emission_prob[(sentence[x], tag)])
                    temp[tag].append(value)
                else:
                    temp_prob = -999999999
                    prev_tag = 'TEMP'
                    #print(x)
                    #print(trellis[sentence[x-1]][0]['DET'])
                    for tag2 in tag_counter:
                        prev_prob = trellis[sentence[x-1]][0][tag2][0] + numpy.log(trans_prob[(tag2, tag)])
                        if prev_prob > temp_prob:
                            prev_tag = tag2
                            temp_prob = prev_prob
                    temp[(prev_tag, tag)].append((numpy.log(emission_prob[(sentence[x], tag)]) + temp_prob))
            trellis[sentence[x]].append(temp)
            #print(trellis[sentence[x]][x]['DET'][0])
            first_word = 1
            last_word = sentence[x]
        sent = []
        value_list = trellis[last_word]
        #largest_tag = max(value_list, key=lambda k: value_list[k])


        print(value_list)



    #raise Exception("You must implement me")
    return predicts

def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''


    predicts = []
    #raise Exception("You must implement me")

    #set up dict
    tag_counter = Counter()
    word_dict = defaultdict(list)
    for sentence in train:
        for word, tag in sentence:
            tag_counter[tag] += 1
            word_dict[word].append(tag)

    #set up most tags per word
    word_tag_assign = {}
    for word in word_dict:
        word_tag_counter = Counter()
        for tag in word_dict[word]:
            word_tag_counter[tag] +=1
        word_tag_assign[word] = word_tag_counter.most_common(1)[0][0]

    #assign test tag values
    for sentence in test:
        sent = []
        for word in sentence:
            if word in word_tag_assign:
                sent.append((word, word_tag_assign[word]))
            else:
                sent.append((word, tag_counter.most_common(1)[0][0]))
        predicts.append(sent)

    return predicts
