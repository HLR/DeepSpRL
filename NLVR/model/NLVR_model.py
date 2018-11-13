# -*- coding: utf-8 -*-

'''
Institution: Tulane University
Name: Chen Zheng
Date: 10/23/2018
Purpose: Building up the deep model.
'''

from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
#


class LSTMmodel(nn.Module):
    def __init__(self, vocab_input_size, embedding_size, fea_input_size, hidden_size):
        super(LSTMmodel, self).__init__()
        self.hidden_size_image = hidden_size
        self.hidden_size_sen = hidden_size
        self.embedding = nn.Embedding(vocab_input_size, embedding_size)
        self.lstm1 = nn.LSTM(fea_input_size, hidden_size, num_layers=1)
        self.lstm2 = nn.LSTM(fea_input_size, hidden_size, num_layers=1)
        self.lstm3 = nn.LSTM(fea_input_size, hidden_size, num_layers=1)
        self.lstm4 = nn.LSTM(embedding_size, hidden_size, num_layers=1)
        self.classification = nn.Linear(hidden_size, 2)



    def forward(self, input1, input2, input3, input_sen, input1_len, input2_len,
                input3_len, input_sen_len, batch_size, embed_size, hidden_size):
        # print(input1)
        # print(input1_len.size())
        pack1 = torch.nn.utils.rnn.pack_padded_sequence(input1.view(batch_size, -1, 9), input1_len, batch_first=True)
        # print('--------------',pack1)
        output_1, hidden_1 = self.lstm1(pack1)
        pack2 = torch.nn.utils.rnn.pack_padded_sequence(input2.view(batch_size, -1, 9), input2_len, batch_first=True)
        output_2, hidden_2 = self.lstm2(pack2, hidden_1)
        pack3 = torch.nn.utils.rnn.pack_padded_sequence(input3.view(batch_size, -1, 9), input3_len, batch_first=True)
        output_3, hidden_3 = self.lstm3(pack3, hidden_2)
        # output_3, hidden_3 = torch.nn.utils.rnn.pad_packed_sequence(output_3, batch_first=True)
        # encoder_outputs, _ = torch.nn_utils.rnn.pad_packed_sequence(output_3, batch_first=True)

        embedded = self.embedding(input_sen).view(batch_size, -1, embed_size)
        output_sen = embedded
        pack4 = torch.nn.utils.rnn.pack_padded_sequence(output_sen.view(batch_size, -1, embed_size), input_sen_len, batch_first=True)
        output_sen, hidden_4 = self.lstm4(pack4, hidden_3)
        encoder_outputs, hidden_5 = torch.nn.utils.rnn.pad_packed_sequence(output_sen, batch_first=True)
        # _, _ = torch.nn.utils.rnn.pad_packed_sequence(output_1, batch_first=True)
        # _, _ = torch.nn.utils.rnn.pad_packed_sequence(output_2, batch_first=True)
        # _, _ = torch.nn.utils.rnn.pad_packed_sequence(output_3, batch_first=True)

        # print('0', encoder_outputs.shape, hidden_5)
        # return output_1, output_2, output_3, output_sen, hidden_1, hidden_2, hidden_3, hidden_4
        y_pred = self.classification(encoder_outputs[-1][3])
        # print(encoder_outputs[-1][3].shape)
        # print("y_pred", y_pred)
        return y_pred
        # return output_sen, hidden_4

    def initHidden(self, hidden_size):
        return torch.zeros(1, 1, hidden_size, device=device)

# class LSTMmodel(nn.Module):
#     def __init__(self, vocab_input_size, embedding_size, hidden_size_image, hidden_size_sen):
#         super(LSTMmodel, self).__init__()
#         self.hidden_size_image = hidden_size_image
#         self.hidden_size_sen = hidden_size_sen
#         self.embedding = nn.Embedding(vocab_input_size, embedding_size)
#         self.gru1 = nn.GRU(hidden_size_image, hidden_size_image, num_layers=2)
#         self.gru2 = nn.GRU(hidden_size_image, hidden_size_image, num_layers=2)
#         self.gru3 = nn.GRU(hidden_size_image, hidden_size_image, num_layers=2)
#         self.gru4 = nn.GRU(hidden_size_sen, hidden_size_sen, num_layers=2)
#         self.classification = nn.Linear(hidden_size_sen, 2)
#
#
#
#     def forward(self, input1, input2, input3, input_sen, hidden1, hidden2):
#
#         output_1, hidden_1 = self.gru1(input1, hidden1)
#         output_2, hidden_2 = self.gru2(input2, hidden_1)
#         output_3, hidden_3 = self.gru3(input3, hidden_2)
#
#         embedded = self.embedding(input).view(1, 1, -1)
#         output_sen = embedded
#         output_sen, hidden_4 = self.gru4(output_sen, hidden_3)
#         # return output_1, output_2, output_3, output_sen, hidden_1, hidden_2, hidden_3, hidden_4
#         y_pred = self.classification(hidden_4)
#         return y_pred
#         # return output_sen, hidden_4
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)

# print(LSTMmodel(1000, 128, 64))

'''
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):


class EncoderRNN(nn.Module):
    def __init__(self, vocab_input_size, hidden_size_image, hidden_size_sen):
        super(EncoderRNN, self).__init__()
        self.hidden_size_image = hidden_size_image
        self.hidden_size_sen = hidden_size_sen
        self.embedding = nn.Embedding(vocab_input_size, hidden_size_sen)
        self.gru1 = nn.GRU(hidden_size_image, hidden_size_image)
        self.gru2 = nn.GRU(hidden_size_image, hidden_size_image)
        self.gru3 = nn.GRU(hidden_size_image, hidden_size_image)
        self.gru4 = nn.GRU(hidden_size_sen, hidden_size_sen)



    def forward(self, input1, input2, input3, input_sen, hidden1, hidden2):

        output_1, hidden_1 = self.gru1(input1, hidden1)
        output_2, hidden_2 = self.gru2(input2, hidden1)
        output_3, hidden_3 = self.gru3(input3, hidden1)

        embedded = self.embedding(input).view(1, 1, -1)
        output_sen = embedded
        output_sen, hidden_4 = self.gru4(output_sen, hidden2)
        return output_1, output_2, output_3, output_sen, hidden_1, hidden_2, hidden_3, hidden_4

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

'''
