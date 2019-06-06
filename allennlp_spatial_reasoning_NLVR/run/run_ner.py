from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

import adabound

import sys
sys.path.append('../')
from data_preprocessing.Pos_Reader import PosDatasetReader
from data_preprocessing.Nlvr_Reader import NLVR_DatasetReader
from model.Spatial_NER import LstmTagger
torch.manual_seed(1)

EMBEDDING_DIM = 16
HIDDEN_DIM = 16  # ner 16, other 32
BATCH_SIZE = 16
lr = 0.01

def running_NER():
    reader = PosDatasetReader()
    train_dataset = reader.read('../data/700_multi_data/600_ner_train.txt')
    validation_dataset = reader.read('../data/700_multi_data/66_ner_test.txt')

    vocab = Vocabulary.from_files("../model_store/vocabulary")

    # '''vocab part'''
    # train_1 = reader.read('../data/train/train.json')
    # train_2 = reader.read('../data/train/dev.json')

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
    model = LstmTagger(word_embeddings, lstm, vocab)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=1000)
    trainer.train()

# running_NER()

def test_case(test_sentence):
    reader = PosDatasetReader()
    vocab = Vocabulary.from_files("../model_store/vocabulary")

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    model = LstmTagger(word_embeddings, lstm, vocab)

    with open("../model_store/model.th", 'rb') as f:
        model.load_state_dict(torch.load(f))
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)

    tag_logits = predictor.predict(test_sentence)['tag_logits']
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

test_sentence = "There is a black square touching circle"
test_case(test_sentence)






def generate_res_file():
    reader = PosDatasetReader()
    vocab = Vocabulary.from_files("../model_store/vocabulary")

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    model2 = LstmTagger(word_embeddings, lstm, vocab)

    with open("../model_store/model.th", 'rb') as f:
        model2.load_state_dict(torch.load(f))
    predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)

    train_read_file = open('../data/only_sentence/raw_test.json', 'r')
    train_write_file = open('../data/only_sentence/ner_test.json', 'w')
    for line in train_read_file:
        tag_logits2 = predictor2.predict(line.replace('.', '').replace(',', '').replace('\n', ''))['tag_logits']
        tag_ids = np.argmax(tag_logits2, axis=-1)
        res = [model2.vocab.get_token_from_index(i, 'labels') for i in tag_ids]
        for i in range(len(res)):
            train_write_file.write(res[i]+' ')
        # train_write_file.write(str(tag_logits2))
        train_write_file.write('\n')
        train_write_file.flush()
    train_read_file.close()
    train_write_file.close()
    print('finish')
# generate_res_file()