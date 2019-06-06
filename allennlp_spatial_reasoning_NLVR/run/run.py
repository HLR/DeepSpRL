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
from model.Property_generating import Property_generating
from model.All_generating import All_generating
torch.manual_seed(1)

EMBEDDING_DIM = 16
HIDDEN_DIM = 32
BATCH_SIZE = 16
lr = 0.01

reader = NLVR_DatasetReader()
whole_train_dataset = reader.read(['../data/train/train.json', '../data/only_sentence/ner_train.json'])
whole_validation_dataset = reader.read(['../data/test/test.json', '../data/only_sentence/ner_test.json'])
vocab = Vocabulary.from_instances(whole_train_dataset + whole_validation_dataset)


def running_whole_model():
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    iterator = BucketIterator(batch_size=BATCH_SIZE, sorting_keys=[("sentence", "num_tokens"),
                                                                   ("structures1", "num_tokens"),
                                                                   ("structures2", "num_tokens"),
                                                                   ("structures3", "num_tokens")])
    iterator.index_with(vocab)


    model = All_generating(embed_size=EMBEDDING_DIM,
                           word_embeddings=word_embeddings,
                           vocab=vocab,
                           num_of_candidates=7,
                           )

    # optimizer = adabound.AdaBound(model.parameters(), lr=lr, final_lr=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)


    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=whole_train_dataset,
                      validation_dataset=whole_validation_dataset,
                      patience=5,
                      num_epochs=30)
    trainer.train()

running_whole_model()

