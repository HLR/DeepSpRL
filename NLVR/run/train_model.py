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
from data_helper.data_helper import read_file, preprocess_sentence, image_feature_tensor, sentence_to_tensor
from train_test_functions.train_funcs import trainIters
from train_test_functions.test_funcs import testIters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(CONFIG['MAX_LENGTH'])

'''
Read the training data
'''
input_data, sentences, label = read_file(CONFIG['TRAIN_DIR'])
# print(training_data)
word2index, word2count, index2word, n_words = preprocess_sentence(sentences)
print(n_words)

input_0, input_1, input_2, input_0_len, input_1_len, input_2_len, target = image_feature_tensor(input_data,
                                                                            label, CONFIG['feature_length'])
input_tensor, input_length = sentence_to_tensor(word2index, sentences, CONFIG['MAX_LENGTH'])

model = LSTMmodel(n_words, CONFIG['embed_size'], 9, CONFIG['hidden_size']).to(device)
print(model)

input_data_test, sentences_test, label_test = read_file(CONFIG['TEST_DIR'])

input_0_test, input_1_test, input_2_test, input_0_len_test, input_1_len_test, \
    input_2_len_test, target_test = image_feature_tensor(input_data_test, label_test, CONFIG['feature_length'])
input_tensor_test, input_length_test = sentence_to_tensor(word2index, sentences_test, CONFIG['MAX_LENGTH'])


trainIters(input_0, input_1, input_2, input_tensor, input_0_len, input_1_len,
           input_2_len, input_length, target, model, CONFIG['hidden_size'],
            input_0_test, input_1_test, input_2_test, input_tensor_test, input_0_len_test, input_1_len_test,
            input_2_len_test, input_length_test, target_test
           )

testIters(input_0_test, input_1_test, input_2_test, input_tensor_test, input_0_len_test, input_1_len_test,
          input_2_len_test, input_length_test, target_test, model, CONFIG['hidden_size'])

######################################################################
#

# evaluateRandomly(encoder1, attn_decoder1)


######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

# S


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#
'''
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")

'''
