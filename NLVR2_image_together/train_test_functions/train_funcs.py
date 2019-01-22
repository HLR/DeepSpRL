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


def begin_to_train(input1, input2, input3, input_total, input_sen, input1_len, input2_len, input3_len,
                   input_total_len, input_sen_len, target, model, optimizer, criterion, hidden_size):
    # hidden_tensor = model.initHidden(hidden_size)
    optimizer.zero_grad()
    # input_length = input_sen.size(0)

    y_pred = model(input1, input2, input3, input_total, input_sen, input1_len, input2_len, input3_len,
                   input_total_len, input_sen_len, CONFIG['batch_size'], CONFIG['embed_size'], CONFIG['hidden_size'])

    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == target.view(-1)).sum().item()
    #print("correct = ", correct)

    loss = criterion(y_pred, target.view(-1))
    # print(y_pred.size(), target.view(-1).size())

    loss.backward()
    optimizer.step()
    print("correct = ", correct, ' loss=', loss.item())
    return loss.item(), correct


def trainIters(input1, input2, input3, input_total, input_sen, input1_len, input2_len, input3_len,
               input_total_len, input_sen_len, target, model, hidden_size,
               input_0_test, input_1_test, input_2_test, input_total_test, input_tensor_test,
               input_0_len_test, input_1_len_test, input_2_len_test, input_total_len_test, input_length_test, target_test
               ):
    # start = time.time()
    print_loss_total = 0  # Reset every print_every
    print_acc_total = 0
    # plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    f_train = open(CONFIG['save_train_result_dir'], 'w')
    f_test = open(CONFIG['save_test_result_dir'], 'w')

    for key, value in CONFIG.items():
        f_train.write(str(key) + ' : ' + str(value) + '\n')
        f_test.write(str(key) + ' : ' + str(value) + '\n')

    for iter in range(1, CONFIG['n_iters'] + 1):
        bad_count = 0
        print('it is the ', iter, 'iteration')
        for i in range(0, input1.size()[0], CONFIG['batch_size']):
            # print('----->',input3_len[i])
            # print('come here')
            # print(len(input1_len[i:i+CONFIG['batch_size']]), input1_len[i:i+CONFIG['batch_size']])
            # print(len(input2_len[i:i + CONFIG['batch_size']]), input2_len[i:i + CONFIG['batch_size']])
            # print(len(input3_len[i:i + CONFIG['batch_size']]), input3_len[i:i + CONFIG['batch_size']])
            # print('-------------------------------------------------------------')


            # loss, correct = begin_to_train(input1[i:i+CONFIG['batch_size']],
            #                                input2[i:i+CONFIG['batch_size']],
            #                                input3[i:i+CONFIG['batch_size']],
            #                                input_total[i:i+CONFIG['batch_size']],
            #                                input_sen[i:i+CONFIG['batch_size']],
            #                                input1_len[i:i+CONFIG['batch_size']],
            #                                input2_len[i:i+CONFIG['batch_size']],
            #                                input3_len[i:i+CONFIG['batch_size']],
            #                                input_total_len[i:i+CONFIG['batch_size']],
            #                                input_sen_len[i:i+CONFIG['batch_size']],
            #                                target[i:i+CONFIG['batch_size']],
            #                                model, optimizer, criterion, hidden_size)
            # # loss, correct = begin_to_train(input1[i], input2[i], input3[i], input_sen[i], input1_len[i], input2_len[i],
            # #                                input3_len[i], input_sen_len[i], target[i], model, optimizer, criterion, hidden_size)
            #
            # print_loss_total += loss
            # print_acc_total += correct

            try:
                loss, correct = begin_to_train(input1[i:i + CONFIG['batch_size']],
                                               input2[i:i + CONFIG['batch_size']],
                                               input3[i:i + CONFIG['batch_size']],
                                               input_total[i:i + CONFIG['batch_size']],
                                               input_sen[i:i + CONFIG['batch_size']],
                                               input1_len[i:i + CONFIG['batch_size']],
                                               input2_len[i:i + CONFIG['batch_size']],
                                               input3_len[i:i + CONFIG['batch_size']],
                                               input_total_len[i:i + CONFIG['batch_size']],
                                               input_sen_len[i:i + CONFIG['batch_size']],
                                               target[i:i + CONFIG['batch_size']],
                                               model, optimizer, criterion, hidden_size)
                print_loss_total += loss
                print_acc_total += correct
            except:
                # print('the', i, 'th data has problem')
                bad_count += 1
                pass


            # if (iter*(input1.size()[0])+i) % CONFIG['print_every'] == 0:
            #     print_loss_avg = print_loss_total / CONFIG['print_every']
            #     print_loss_total = 0
            #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / CONFIG['n_iters']),
            #                                  iter, iter / CONFIG['n_iters'] * 100, print_loss_avg))
        if iter % 1 == 0:
            '''
            train part
            '''
            print_loss_avg = float(print_loss_total) / (input1_len.size()[0] // CONFIG['batch_size'])
            print('training acc is: ', float(print_acc_total) / (input1_len.size()[0] - bad_count), ', training loss is: ', print_loss_avg,
                  ', total training size is: ', (input1_len.size()[0] - bad_count))

            f_train.write('training acc is: ' + str(float(print_acc_total) / (input1_len.size()[0] - bad_count)) +
                          ', training loss is: ' + str(print_loss_avg) +
                          ', total training size is: ' + str((input1_len.size()[0] - bad_count)) + '\n')
            f_train.flush()
            print_acc_total = 0
            print_loss_total = 0
            '''
            test part
            '''
            test_res = testIters(input_0_test, input_1_test, input_2_test, input_total_test, input_tensor_test, input_0_len_test, input_1_len_test,
                      input_2_len_test, input_total_len_test, input_length_test, target_test, model, CONFIG['hidden_size'])
            f_test.write(test_res)
            f_test.flush()

    # after training, save  model
    f_train.close()
    f_test.close()
    torch.save(model.state_dict(), CONFIG['save_checkpoint_dir'])
    # model.save_state_dict(CONFIG['save_checkpoint_dir'])

    # load  previously training model:
    # model.load_state_dict(torch.load('mytraining.pt'))

