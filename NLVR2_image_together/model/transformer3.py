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
This python file implements transformer-transformer-matching model.
Sentence use self-attn, which q == k == v. input is embedding of each word in sentence(n, embed_size),
and the output is the same dimension (n, embed_size), which n is the length of sentence.
Structure representation use attn, which q is the output of Sentence (n, embed_size) by the above operation,
and k == v is the structure representation which input is (m, 9) --> (m, embed_size) by a linear layer,
and the output is is the same dimension (n, embed_size), which n is the length of sentence, m is the number
of objects in the subimages.
and then compute the similarity between (n, embed_size) and (n, embed_size), and then choose true or false.
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
        # print(q.size(), k.size(), v.size())
        sz_b, len_q, _,  = q.size()
        sz_b, len_k, _,  = k.size()
        sz_b, len_v, _,  = v.size()


        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        # output, attn = self.attention(q, k, v, mask=mask)

        output, attn = self.attention(q, k, v)

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

    def forward(self, q_input, k_input, v_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            q_input, k_input, v_input, mask=slf_attn_mask)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn



class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, q_embed, k_embed, v_embed, return_attns=True):

        enc_slf_attn_list = []

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                q_embed, k_embed, v_embed
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,



class Transformer3(nn.Module):
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
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)


        '''
        Transformer model for image
        '''

        self.linear_image = torch.nn.Linear(9, self.embedding_size)

        self.encoder_image = Encoder(
            d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        '''
        Final classification
        '''
        self.classification = nn.Linear(CONFIG['TOPK'], 2)


    def forward(self, input1, input2, input3, input_total, input_sen, input1_len, input2_len,
                input3_len, input_total_len, input_sen_len, batch_size, embed_size, hidden_size):
        '''sentence'''
        sen_emb = self.embedding(input_sen).view(batch_size, -1, self.embedding_size)

        # input_sen = input_sen.view(CONFIG['batch_size'], -1)
        enc_output_sen, *_ = self.encoder_question(sen_emb, sen_emb, sen_emb)
        # print('enc_output_sen.size()', enc_output_sen.size())

        '''image'''
        input1_linear = self.linear_image(input1).view(-1, CONFIG['feature_length'], self.embedding_size)
        enc_output_img1, *_ = self.encoder_image(sen_emb, input1_linear, input1_linear)

        input2_linear = self.linear_image(input2).view(-1, CONFIG['feature_length'], self.embedding_size)
        enc_output_img2, *_ = self.encoder_image(sen_emb, input2_linear, input2_linear)

        input3_linear = self.linear_image(input3).view(-1, CONFIG['feature_length'], self.embedding_size)
        enc_output_img3, *_ = self.encoder_image(sen_emb, input3_linear, input3_linear)

        image_output = torch.cat([enc_output_img1, enc_output_img2, enc_output_img3], dim=1)
        '''
        compute the matching matrix
        '''
        matrix = torch.matmul(image_output, torch.transpose(enc_output_sen, 2, 1))

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
