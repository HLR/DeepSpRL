'''
Institution: Tulane University
Name: Chen Zheng
Date: 11/04/2018
Purpose: Some functions help to test process.
'''
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
sys.path.append('../')
from config.first_config import CONFIG
import time
import math

def begin_to_test(input1, input2, input3, input_sen, input1_len, input2_len, input3_len, input_sen_len,
                   target, model, hidden_size):
    # hidden_tensor = model.initHidden(hidden_size)
    #
    # # load  previously training model:
    # model.load_state_dict(torch.load(CONFIG['save_checkpoint_dir']))

    y_pred = model(input1, input2, input3, input_sen, input1_len, input2_len, input3_len, input_sen_len,
                   CONFIG['batch_size'], CONFIG['embed_size'], CONFIG['hidden_size'])

    # print('2', y_pred.view(-1,2))
    # # values, indices = torch.max(y_pred, 0)
    # # print('3', indices.view(-1))
    # print('4', target)

    # print(y_pred.view(-1, 2))
    # print('---------------')
    # print(target)
    values, indices = torch.max(y_pred, 0)
    # print(indices)
    correct = (indices == target).sum()
    # print(correct)
    # accuracy = 100 * correct / total
    # print(accuracy)

    return correct



def testIters(input1, input2, input3, input_sen, input1_len, input2_len, input3_len, input_sen_len,
               target, model, hidden_size):
    total_acc = 0


    for i in range(input1_len.size()[0]):
        # print('----->',input3_len[i])
        try:
            acc = begin_to_test(input1[i], input2[i], input3[i], input_sen[i], input1_len[i], input2_len[i],
                                  input3_len[i], input_sen_len[i], target[i], model,
                                  hidden_size)
            total_acc += acc
        except:
            print('the', i, 'th data has problem')

    print(total_acc)
    print(input1_len.size()[0])
    print(float(total_acc) / input1_len.size()[0])

    f = open(CONFIG['save_test_result_dir'], 'w')
    f.write(str(float(total_acc) / input1_len.size()[0]))
    f.close()



######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')