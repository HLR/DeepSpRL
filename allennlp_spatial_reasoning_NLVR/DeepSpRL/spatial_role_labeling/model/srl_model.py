# -*- coding:utf8 -*-
'''
Institution: Tulane University
Name: Chen Zheng
Date: 10/27/2018
Purpose: Building up the deep model.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
sys.path.append('../')
from config.first_config import CONFIG


class sprl_model(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim, vocab_size, pre_emb, candiate_BIO_number):
        super(sprl_model, self).__init__()

        self.hidden_dim = hidden_dim
        self.candiate_BIO_number = candiate_BIO_number
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(pre_emb))
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.candiate_BIO_number)

    def forward(self, input_sen, sentence_len, relation_vector, total_shape, y_batch,
                batch_size, embed_size):

        embedded = self.embedding(input_sen).view(batch_size, -1, embed_size)
        inputs = torch.cat([embedded, relation_vector], 2)
        pack = torch.nn.utils.rnn.pack_padded_sequence(inputs.view(batch_size, -1, embed_size), sentence_len,
                                                        batch_first=True)
        output_sen, hidden_before_pad = self.lstm(pack)
        lstm_output, hidden_after_pad = torch.nn.utils.rnn.pad_packed_sequence(output_sen, batch_first=True)
        hidden2tag = nn.Linear(lstm_output)


        # self.char_lstm_hidden = self.init_lstm_hidden(dim=self.char_lstm_dim, bidirection=True, batchsize=chars2.size(0))
        chars_embeds = self.char_embeds(chars2).transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars2_length)
        lstm_out, _ = self.char_lstm(packed)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        outputs = outputs.transpose(0, 1)
        chars_embeds_temp = Variable(torch.FloatTensor(torch.zeros((outputs.size(0), outputs.size(2)))))
        if self.use_gpu:
            chars_embeds_temp = chars_embeds_temp.cuda()
        for i, index in enumerate(output_lengths):
            chars_embeds_temp[i] = torch.cat(
                (outputs[i, index - 1, :self.char_lstm_dim], outputs[i, 0, self.char_lstm_dim:]))
        chars_embeds = chars_embeds_temp.clone()
        for i in range(chars_embeds.size(0)):
            chars_embeds[d[i]] = chars_embeds_temp[i]



        embeds = self.word_embeds(sentence)
        if self.n_cap and self.cap_embedding_dim:
            cap_embedding = self.cap_embeds(caps)

        if self.n_cap and self.cap_embedding_dim:
            embeds = torch.cat((embeds, chars_embeds, cap_embedding), 1)
        else:
            embeds = torch.cat((embeds, chars_embeds), 1)

        embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim * 2)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    #
    # def viterbi_decode(self, feats):
    #     backpointers = []
    #     # analogous to forward
    #     init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
    #     init_vvars[0][self.tag_to_ix[START_TAG]] = 0
    #     forward_var = Variable(init_vvars)
    #     if self.use_gpu:
    #         forward_var = forward_var.cuda()
    #     for feat in feats:
    #         next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
    #         _, bptrs_t = torch.max(next_tag_var, dim=1)
    #         bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
    #         next_tag_var = next_tag_var.data.cpu().numpy()
    #         viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
    #         viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
    #         if self.use_gpu:
    #             viterbivars_t = viterbivars_t.cuda()
    #         forward_var = viterbivars_t + feat
    #         backpointers.append(bptrs_t)
    #
    #     terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    #     terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
    #     terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
    #     best_tag_id = argmax(terminal_var.unsqueeze(0))
    #     path_score = terminal_var[best_tag_id]
    #     best_path = [best_tag_id]
    #     for bptrs_t in reversed(backpointers):
    #         best_tag_id = bptrs_t[best_tag_id]
    #         best_path.append(best_tag_id)
    #     start = best_path.pop()
    #     assert start == self.tag_to_ix[START_TAG]
    #     best_path.reverse()
    #     return path_score, best_path





