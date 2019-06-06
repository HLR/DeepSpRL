import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config.first_config import CONFIG

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

''' Tips by Chen Zheng
This python file implements transformer model.
In the framework of Google transformer model, it has encoder and decoder part.
In encoder part, it has self attention and Position wise Feed Forward;
In decoder part, it has self attention and Position wise Feed Forward and encoder attention;
Self attention is equal to MultiHeadAttention when k == v
MultiHeadAttention has ScaledDotProductAttention, layer_norm and fc.

'''


'''
ScaledDotProductAttention, Multiheadlayer and PositionwiseFeedForward
'''

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        # if mask is not None:
        #     attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # print(d_model, n_head, d_k, n_head * d_k)
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _,  = q.size()
        sz_b, len_k, _,  = k.size()
        sz_b, len_v, _,  = v.size()


        residual = q

        # print("q.size():", q.size())
        # # print('self.w_qs(q).size(): ', self.w_qs(q).size())
        # q = torch.cat([q]*n_head, 1).view(sz_b, len_q, n_head, d_k)
        # print("q.size():", q.size())
        # print('self.w_qs(q).size(): ', self.w_qs(q).size())
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output



'''
Encoder layer and Decoder Layer.
'''

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn




def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1,  len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=PAD)

        # self.position_enc = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
        #     freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, return_attns=False):

        enc_slf_attn_list = []
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        # enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        enc_output = self.src_word_emb(src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=PAD)

        # self.position_enc = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
        #     freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        # dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
        dec_output = self.tgt_word_emb(tgt_seq)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,



class Transformer(nn.Module):
    def __init__(
            self, vocab_input_size, fea_input_size,
            d_model=CONFIG['embed_size'], d_inner=2048, n_layers=6, n_head=8, d_k=CONFIG['embed_size']//8, d_v=CONFIG['embed_size']//8, dropout=0.1):

        super().__init__()

        self.vocab_input_size = vocab_input_size
        self.embedding_size = CONFIG['embed_size']
        self.fea_input_size = fea_input_size
        self.hidden_size = CONFIG['hidden_size']
        self.batch_size = CONFIG['batch_size']
        self.embedding = nn.Embedding(self.vocab_input_size, self.embedding_size)

        '''
        Transformer model for sentence
        '''
        self.encoder_question = Encoder(
            n_src_vocab=vocab_input_size, len_max_seq=CONFIG['MAX_LENGTH'],
            d_word_vec=self.embedding_size, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder_question = Decoder(
            n_tgt_vocab=vocab_input_size, len_max_seq=CONFIG['MAX_LENGTH'],
            d_word_vec=self.embedding_size, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.lstm1 = nn.RNN(fea_input_size, self.embedding_size, num_layers=CONFIG['lstm_num_layer'], nonlinearity='relu')
        '''
        Custom Lstm
        '''
        # self.nlayers = 1
        # self.dropout = nn.Dropout(p=0.1)
        #
        # ih, hh = [], []
        # for i in range(self.nlayers):
        #     ih.append(nn.Linear(self.fea_input_size, 4 * self.hidden_size))
        #     hh.append(nn.Linear(self.hidden_size, 4 * self.hidden_size))
        # self.w_ih = nn.ModuleList(ih)
        # self.w_hh = nn.ModuleList(hh)

        '''
        Final classification
        '''
        self.classification = nn.Linear(CONFIG['TOPK'], 2)

        # self.tgt_word_prj = nn.Linear(d_model, self.vocab_input_size, bias=False)
        # nn.init.xavier_normal_(self.tgt_word_prj.weight)
        #
        # 'To facilitate the residual connections, \
        #  the dimensions of all module outputs shall be the same.'
        #
        # # Share the weight matrix between target word embedding & the final logit dense layer
        # self.tgt_word_prj.weight = self.decoder_question.tgt_word_emb.weight
        # self.x_logit_scale = (d_model ** -0.5)
        #
        # "To share word embedding table, the vocabulary size of src/tgt shall be the same."
        # self.encoder_question.src_word_emb.weight = self.decoder_question.tgt_word_emb.weight

    def forward(self, input1, input2, input3, input_total, input_sen, input1_len, input2_len,
                input3_len, input_total_len, input_sen_len, batch_size, embed_size, hidden_size):

        # embedded = self.embedding(input_sen).view(self.batch_size, -1, self.embedding_size)
        input_sen = input_sen.view(batch_size, -1)
        enc_output, *_ = self.encoder_question(input_sen)
        # enc_output[enc_output != enc_output] = 0
        # print('enc_output.size(): ', enc_output.size())
        dec_output, *_ = self.decoder_question(input_sen, input_sen, enc_output)
        # dec_output[dec_output != dec_output] = 0

        # print('dec_output.size(): ', dec_output.size())
        # sentence_output = torch.transpose(dec_output, 0, 1)
        sentence_output = dec_output
        # sentence_output = sentence_output[:, :, :]
        # sentence_output = sentence_output[:, 0: input_sen_len, :]
        print('sentence_output', sentence_output)




        # pack1 = torch.nn.utils.rnn.pack_padded_sequence(input1.view(batch_size, -1, 9), input1_len, batch_first=True)
        # output_1, hidden_1 = self.lstm1(pack1)
        # output_1, len_1 = torch.nn.utils.rnn.pad_packed_sequence(output_1, batch_first=True)
        # # print('output_1.size(): ', output_1.size())
        #
        # pack2 = torch.nn.utils.rnn.pack_padded_sequence(input2.view(batch_size, -1, 9), input2_len, batch_first=True)
        # output_2, hidden_2 = self.lstm1(pack2, hidden_1)
        # output_2, len_2 = torch.nn.utils.rnn.pad_packed_sequence(output_2, batch_first=True)
        # # print('output_2.size(): ', output_2.size())
        #
        # pack3 = torch.nn.utils.rnn.pack_padded_sequence(input3.view(batch_size, -1, 9), input3_len, batch_first=True)
        # output_3, hidden_3 = self.lstm1(pack3, hidden_2)
        # output_3, len_3 = torch.nn.utils.rnn.pad_packed_sequence(output_3, batch_first=True)
        # # print('output_3.size(): ', output_3.size())
        # # print('-------------------------------------------------------')

        # output_1, hidden_1 = self.lstm1(input1.view(batch_size, -1, 9))
        # output_2, hidden_2 = self.lstm1(input2.view(batch_size, -1, 9))
        # output_3, hidden_3 = self.lstm1(input3.view(batch_size, -1, 9))
        # # print(input1.view(batch_size, -1, 9).size(), input2.view(batch_size, -1, 9).size(), input3.view(batch_size, -1, 9).size())
        # # print(input1_len.size(), input2_len.size(), input3_len.size())
        # output_1 = output_1[:, :, :]
        # output_2 = output_2[:, :, :]
        # output_3 = output_3[:, :, :]
        # image_output = torch.cat([output_1, output_2, output_3], dim=1)
        # print('image_output.size():', image_output.size())
        # print('-------------------------------------------------------')

        '''
        sharing the lstm layer
        '''
        image_output, hidden_1 = self.lstm1(input_total.view(batch_size, -1, 9))
        print('image_output: ', image_output)

        '''
        Custom sharing the lstm layer
        '''
        # hy, cy = [], []
        # for i in range(self.nlayers):
        #     hx, cx = hidden[0][i], hidden[1][i]
        #     gates = self.w_ih[i](input) + self.w_hh[i](hx)
        #     i_gate, f_gate, c_gate, o_gate = gates.chunk(4, 1)
        #
        #     i_gate = F.sigmoid(i_gate)
        #     f_gate = F.sigmoid(f_gate)
        #     c_gate = F.tanh(c_gate)
        #     o_gate = F.sigmoid(o_gate)
        #
        #     ncx = (f_gate * cx) + (i_gate * c_gate)
        #     nhx = o_gate * F.tanh(ncx)
        #     cy.append(ncx)
        #     hy.append(nhx)
        #     input = self.dropout(nhx)
        #
        # image_output, cy = torch.stack(hy, 0), torch.stack(cy, 0)
        # print('image_output.size(): ', image_output.size())



        # print('image_output: ', image_output.size(), image_output)
        # print('image_output[:, 0: 10, :]: ', image_output[:, 0: 10, :].size(), image_output[:, 0: 10, :])
        # print('image_output[:, 10: 20, :]: ', image_output[:, 10: 20, :].size(), image_output[:, 10: 20, :])
        # print('image_output[:, 20: 30, :]: ', image_output[:, 20: 30, :].size(), image_output[:, 20: 30, :])


        '''
        compute the matching matrix
        '''
        matrix = torch.matmul(image_output, torch.transpose(sentence_output, 2, 1))
        # print('matrix: ', matrix.size())

        '''
        CNN will have problem, so i change my mind to extract the top-k element based on my matrix.
        '''
        matrix_2_tensor = matrix.view(batch_size, -1)
        # print('matrix_2_tensor: ', matrix_2_tensor.size())
        top_k = torch.topk(matrix_2_tensor, CONFIG['TOPK'])[0]
        # print('top_k.shape: ', top_k.shape, ', top_k: ', top_k)
        y_pred = self.classification(top_k)
        # print("y_pred", y_pred[0])
        # print('y_pred.size: ', y_pred.size(), ', y_pred:', y_pred)
        # print('-----------------------------------------------')
        # return y_pred[0]
        return y_pred