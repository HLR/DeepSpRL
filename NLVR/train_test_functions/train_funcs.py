'''
Institution: Tulane University
Name: Chen Zheng
Date: 10/23/2018
Purpose: Some functions help to train process.
'''

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import sys
sys.path.append('../')
from config.first_config import CONFIG
from train_test_functions.test_funcs import testIters
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def begin_to_train(input1, input2, input3, input_sen, input1_len, input2_len, input3_len, input_sen_len,
                   target, model, optimizer, criterion, hidden_size):
    # hidden_tensor = model.initHidden(hidden_size)
    optimizer.zero_grad()
    # input_length = input_sen.size(0)

    y_pred = model(input1, input2, input3, input_sen, input1_len, input2_len, input3_len, input_sen_len,
                   CONFIG['batch_size'], CONFIG['embed_size'], CONFIG['hidden_size'])

    # print('2', y_pred.view(-1,2))
    # # values, indices = torch.max(y_pred, 0)
    # # print('3', indices.view(-1))
    # print('4', target)
    loss = criterion(y_pred.view(-1, 2), target)
    loss.backward()
    optimizer.step()
    return loss


def trainIters(input1, input2, input3, input_sen, input1_len, input2_len, input3_len, input_sen_len,
               target, model, hidden_size,
               input_0_test, input_1_test, input_2_test, input_tensor_test, input_0_len_test, input_1_len_test,
               input_2_len_test, input_length_test, target_test
               ):
    start = time.time()
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    for iter in range(1, CONFIG['n_iters'] + 1):
        print('it is the ', iter, 'iteration')
        for i in range(input1_len.size()[0]):
            # print('----->',input3_len[i])
            try:
                loss = begin_to_train(input1[i], input2[i], input3[i], input_sen[i], input1_len[i], input2_len[i],
                                      input3_len[i], input_sen_len[i], target[i], model, optimizer, criterion, hidden_size)
                print_loss_total += loss
            except:
                # print('the', i, 'th data has problem')
                pass


            # if (iter*(input1.size()[0])+i) % CONFIG['print_every'] == 0:
            #     print_loss_avg = print_loss_total / CONFIG['print_every']
            #     print_loss_total = 0
            #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / CONFIG['n_iters']),
            #                                  iter, iter / CONFIG['n_iters'] * 100, print_loss_avg))
        if iter % 10 == 0:
            testIters(input_0_test, input_1_test, input_2_test, input_tensor_test, input_0_len_test, input_1_len_test,
                      input_2_len_test, input_length_test, target_test, model, CONFIG['hidden_size'])

    # after training, save  model
    torch.save(model.state_dict(), CONFIG['save_checkpoint_dir'])
    # model.save_state_dict(CONFIG['save_checkpoint_dir'])

    # load  previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))

