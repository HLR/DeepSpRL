# -*- coding: utf-8 -*-

'''
Institution: Tulane University
Name: Chen Zheng
Date: 10/23/2018
Purpose: Running the model.
'''
from __future__ import unicode_literals, print_function, division
import torch
import sys
sys.path.append('../')
from config.first_config import CONFIG
from model.NLVR_model import LSTMmodel
from model.matching_model import matching_model
from model.Bi_matching_model import bi_matching_model
from model.Transformer_model import Transformer
from data_helper.data_helper import read_file, preprocess_sentence, image_feature_tensor, sentence_to_tensor
from train_test_functions.train_funcs import trainIters
from train_test_functions.test_funcs import testIters

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(CONFIG['MAX_LENGTH'])

'''
Read the training data
'''
input_data, sentences, label = read_file(CONFIG['TRAIN_DIR'])
# print(training_data)
word2index, word2count, index2word, n_words = preprocess_sentence(sentences)
print(n_words)

input_0, input_1, input_2, input_total, input_0_len, input_1_len, input_2_len, input_total_len, target = \
                                        image_feature_tensor(input_data, label, CONFIG['feature_length'])
input_tensor, input_length = sentence_to_tensor(word2index, sentences, CONFIG['MAX_LENGTH'])


if CONFIG['MODEL'] == 'NLVR':
    # model = LSTMmodel(n_words, CONFIG['embed_size'], 9, CONFIG['hidden_size']).to(device)
    model = LSTMmodel(n_words, CONFIG['embed_size'], 9, CONFIG['hidden_size'])
    print(model)
elif CONFIG['MODEL'] == 'MATCHING':
    model = matching_model(n_words, CONFIG['embed_size'], 9, CONFIG['hidden_size'])
    print(model)
    # model(input_0[4], input_1[4], input_2[4], input_tensor[4], input_0_len[4], input_1_len[4], input_2_len[4], input_length[4],
    #       CONFIG['batch_size'], CONFIG['embed_size'], CONFIG['hidden_size'])
elif CONFIG['MODEL'] == 'BI-MATCHING':
    model = bi_matching_model(n_words, CONFIG['embed_size'], 9, CONFIG['hidden_size'])
    print(model)
    # model(input_0[4], input_1[4], input_2[4], input_tensor[4], input_0_len[4], input_1_len[4], input_2_len[4], input_length[4],
    #       CONFIG['batch_size'], CONFIG['embed_size'], CONFIG['hidden_size'])
elif CONFIG['MODEL'] == 'TRANSFORMER':
    model = Transformer(n_words, 9)
    # print(model)


input_data_test, sentences_test, label_test = read_file(CONFIG['TEST_DIR'])

input_0_test, input_1_test, input_2_test, input_total_test, input_0_len_test, input_1_len_test, \
    input_2_len_test, input_total_len_test, target_test = image_feature_tensor(input_data_test, label_test, CONFIG['feature_length'])
input_tensor_test, input_length_test = sentence_to_tensor(word2index, sentences_test, CONFIG['MAX_LENGTH'])


trainIters(input_0, input_1, input_2, input_total, input_tensor, input_0_len, input_1_len,
           input_2_len, input_total_len, input_length, target, model, CONFIG['hidden_size'],
            input_0_test, input_1_test, input_2_test, input_total_test, input_tensor_test, input_0_len_test, input_1_len_test,
            input_2_len_test, input_total_len_test, input_length_test, target_test
           )

testIters(input_0_test, input_1_test, input_2_test, input_total_test, input_tensor_test, input_0_len_test, input_1_len_test,
          input_2_len_test, input_total_len_test, input_length_test, target_test, model, CONFIG['hidden_size'])

