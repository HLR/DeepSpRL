# -*- coding: utf-8 -*-

'''
Institution: Tulane University
Name: Chen Zheng
Date: 11/24/2018
Purpose: Building up the matching model.
'''

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import sys
sys.path.append('../')
from config.first_config import CONFIG

class matching_model(nn.Module):
    def __init__(self, vocab_input_size, embedding_size, fea_input_size, hidden_size):
        super(matching_model, self).__init__()
        self.hidden_size_image = hidden_size
        self.hidden_size_sen = hidden_size
        self.embedding = nn.Embedding(vocab_input_size, embedding_size)
        self.lstm1 = nn.LSTM(fea_input_size, hidden_size, num_layers=1)
        self.lstm2 = nn.LSTM(fea_input_size, hidden_size, num_layers=1)
        self.lstm3 = nn.LSTM(fea_input_size, hidden_size, num_layers=1)
        self.lstm4 = nn.LSTM(embedding_size, hidden_size, num_layers=1)
        self.classification = nn.Linear(CONFIG['TOPK'], 2)



    def forward(self, input1, input2, input3, input_sen, input1_len, input2_len,
                input3_len, input_sen_len, batch_size, embed_size, hidden_size):
        '''
        image feature part
        '''
        # print(input_sen)
        # print(input1_len.size())
        pack1 = torch.nn.utils.rnn.pack_padded_sequence(input1.view(batch_size, -1, 9), input1_len, batch_first=True)
        output_1, hidden_1 = self.lstm1(pack1)
        output_1, len_1 = torch.nn.utils.rnn.pad_packed_sequence(output_1, batch_first=True)
        # print('output_1.size(): ', output_1.size())

        pack2 = torch.nn.utils.rnn.pack_padded_sequence(input2.view(batch_size, -1, 9), input2_len, batch_first=True)
        output_2, hidden_2 = self.lstm2(pack2, hidden_1)
        output_2, len_2 = torch.nn.utils.rnn.pad_packed_sequence(output_2, batch_first=True)
        # print('output_2.size(): ', output_2.size())

        pack3 = torch.nn.utils.rnn.pack_padded_sequence(input3.view(batch_size, -1, 9), input3_len, batch_first=True)
        output_3, hidden_3 = self.lstm3(pack3, hidden_2)
        output_3, len_3 = torch.nn.utils.rnn.pad_packed_sequence(output_3, batch_first=True)
        # print('output_3.size(): ', output_3.size())
        image_output = torch.cat([output_1, output_2, output_3], dim=1)
        # print('image_output.size(): ', image_output.size())
        # print('----------------------------------')

        '''
        sentence part
        '''
        embedded = self.embedding(input_sen).view(batch_size, -1, embed_size)
        # print('embedded.size()------->', embedded.size())
        output_sen = embedded
        pack4 = torch.nn.utils.rnn.pack_padded_sequence(output_sen.view(batch_size, -1, embed_size), input_sen_len, batch_first=True)
        output_sen, hidden_4 = self.lstm4(pack4, hidden_3)
        sentence_output, len_4 = torch.nn.utils.rnn.pad_packed_sequence(output_sen, batch_first=True)
        # print('sentence_output.size(): ', sentence_output.size())
        # print('----------------------------------')
        '''
        compute the matching matrix
        '''
        matrix = torch.matmul(image_output, torch.transpose(sentence_output, 2, 1))
        # print('matrix: ', matrix.size())

        '''
        CNN will have problem, so i change my mind to extract the top-k element based on my matrix.
        '''
        matrix_2_tensor = matrix.view(CONFIG['batch_size'], -1)
        # print('matrix_2_tensor: ', matrix_2_tensor.size())
        top_k = torch.topk(matrix_2_tensor, CONFIG['TOPK'])[0]
        # print('top_k: ', top_k.shape)
        y_pred = self.classification(top_k)
        # print("y_pred", y_pred[0])
        return y_pred[0]

