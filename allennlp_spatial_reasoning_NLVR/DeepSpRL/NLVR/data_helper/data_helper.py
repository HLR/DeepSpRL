# -*- coding: utf-8 -*-

'''
Institution: Tulane University
Name: Chen Zheng
Date: 10/23/2018
Purpose: data preprocessing and data clean for the training and testing data.
'''

from __future__ import unicode_literals, print_function, division
import json
from io import open
import unicodedata
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 1
EOS_token = 2

def preprocess_sentence(list):
    '''
    :param list: sentences' list
    :return: the dict pair word and index
    '''

    word2index = {}
    word2count = {}
    index2word = {0: 'MASK', 1: "SOS", 2: "EOS", 3: 'NOT_FOUND'}
    n_words = 4  # Count mask, SOS and EOS, AND NOT_FOUND

    for i in range(len(list)):
        sentence = normalizeString(list[i])
        for word in sentence.split(' '):
            if word not in word2index:
                word2index[word] = n_words
                word2count[word] = 1
                index2word[n_words] = word
                n_words += 1
            else:
                word2count[word] += 1

    return word2index, word2count, index2word, n_words


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# print(normalizeString('a! b? #$%^&*'))


def read_file(filename):
    '''

    :param filename: filename for json format
    :return: 9 vector feature data[x_loc, y_loc, size, shape, type(1,0,0), color(1,0,0)],
             sentence list, and label data(True or False)
    '''
    data_load = []
    feature_data = []
    sentences = []
    label = []
    # '../data/tmp.json'
    for line in open(filename, 'r'):
        data_load.append(json.loads(line))
    # print(data_load[i]['structured_rep'][0][0]['y_loc'])
    # print(len(data_load[0]['structured_rep'][0]))

    # each line of data
    for i in range(len(data_load)): # 100
        # three sub-images
        feature_three_image = []
        for j in range(len(data_load[i]['structured_rep'])): # 3
            feature_each_image = []
            # information in each sub-image
            for k in range(len(data_load[i]['structured_rep'][j])): # x
                feature_each_obj = []
                feature_each_obj.append(data_load[i]['structured_rep'][j][k]['x_loc'])
                feature_each_obj.append(data_load[i]['structured_rep'][j][k]['y_loc'])
                feature_each_obj.append(data_load[i]['structured_rep'][j][k]['size'])
                if data_load[i]['structured_rep'][j][k]['type'] == 'triangle':
                    feature_each_obj.append(1)
                    feature_each_obj.append(0)
                    feature_each_obj.append(0)
                elif data_load[i]['structured_rep'][j][k]['type'] == 'square':
                    feature_each_obj.append(0)
                    feature_each_obj.append(1)
                    feature_each_obj.append(0)
                else:
                    feature_each_obj.append(0)
                    feature_each_obj.append(0)
                    feature_each_obj.append(1)

                if data_load[i]['structured_rep'][j][k]['color'] == 'Yellow':
                    feature_each_obj.append(1)
                    feature_each_obj.append(0)
                    feature_each_obj.append(0)
                elif data_load[i]['structured_rep'][j][k]['color'] == 'Black':
                    feature_each_obj.append(0)
                    feature_each_obj.append(1)
                    feature_each_obj.append(0)
                else:
                    feature_each_obj.append(0)
                    feature_each_obj.append(0)
                    feature_each_obj.append(1)
                # feature_each_obj.append(data_load[i]['structured_rep'][j][k]['color'])

                feature_each_image.append(feature_each_obj)
            feature_three_image.append(feature_each_image)
        # feature_data.append(data_load[i]['sentence'])
        feature_data.append(feature_three_image)
        feature_three_image.append(data_load[i]['sentence'])
        sentences.append(data_load[i]['sentence'])
        if data_load[i]['label'] == 'true':
            label.append(1)
        else:
            label.append(0)


    # for i in range(len(feature_data)):
    #     for j in range(len(feature_three_image)):
    #         print(feature_data[i][j])
    #     # print(feature_data[i][0])
    #     print(label[i])
    #     print('\n')

    return feature_data, sentences, label


# feature_data, sentences, label = read_file('../data/tmp.json')
# word2index, word2count, index2word, n_words = preprocess_sentence(sentences)

# print(word2index)
# print(word2count)
# print(index2word)
# print(n_words)


def sentence_to_tensor(word2index, list, max_length):
    input_list = []
    input_length = []
    for i in range(len(list)):
        sentence = normalizeString(list[i])
        # get(key, default=None)
        # indexes = [word2index[word] for word in sentence.split(' ')]
        indexes = [word2index.get(word, 3) for word in sentence.split(' ')]
        input_length.append(len(indexes))
        # print(indexes)
        if len(indexes) >= max_length:
            indexes = indexes[0: max_length]
        else:
            indexes = indexes + ([0]*(max_length - len(indexes)))
        input_list.append(indexes)

    input_tensor = torch.tensor(input_list, dtype=torch.long, device=device).view(len(list), -1, 1)
    input_length = torch.tensor(input_length, dtype=torch.long, device=device).view(-1, 1)
    return input_tensor, input_length

# input_tensor, input_length = sentence_to_tensor(word2index, sentences, 15)
# print(input_tensor)
# print(input_length)

def image_feature_tensor(feature_data, label, max_length):
    input_0 = []
    input_1 = []
    input_2 = []
    input_0_len = []
    input_1_len = []
    input_2_len = []
    for i in range(len(feature_data)):
        input_0_len.append(len(feature_data[i][0]))
        input_1_len.append(len(feature_data[i][1]))
        input_2_len.append(len(feature_data[i][2]))
        input_0.append(feature_data[i][0] + [[0, 0, 0, 0, 0, 0, 0, 0, 0]]*(max_length-len(feature_data[i][0])))
        input_1.append(feature_data[i][1] + [[0, 0, 0, 0, 0, 0, 0, 0, 0]]*(max_length-len(feature_data[i][1])))
        input_2.append(feature_data[i][2] + [[0, 0, 0, 0, 0, 0, 0, 0, 0]]*(max_length-len(feature_data[i][2])))

    input_0 = torch.tensor(input_0, dtype=torch.float, device=device).view(len(input_0), -1, 9)
    input_1 = torch.tensor(input_0, dtype=torch.float, device=device).view(len(input_1), -1, 9)
    input_2= torch.tensor(input_0, dtype=torch.float, device=device).view(len(input_2), -1, 9)
    input_0_len = torch.tensor(input_0_len, dtype=torch.long, device=device).view(-1, 1)
    input_1_len = torch.tensor(input_1_len, dtype=torch.long, device=device).view(-1, 1)
    input_2_len = torch.tensor(input_2_len, dtype=torch.long, device=device).view(-1, 1)
    label = torch.tensor(label, dtype=torch.long, device=device).view(-1, 1)

    return input_0, input_1, input_2, input_0_len, input_1_len, input_2_len, label


# input_0, input_1, input_2, input_0_len, input_1_len, input_2_len = image_feature_tensor(feature_data, label, 10)
# print(len(input_0))
# print(len(input_0_len))
