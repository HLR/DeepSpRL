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

import sys
sys.path.append('../')
from data_preprocessing.Pos_Reader import PosDatasetReader
torch.manual_seed(1)

class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            self.accuracy(tag_logits, labels, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)

        return output
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


# reader = PosDatasetReader()
# # train_dataset = reader.read(cached_path(
# #     'https://raw.githubusercontent.com/allenai/allennlp'
# #     '/master/tutorials/tagger/training.txt'))
# # validation_dataset = reader.read(cached_path(
# #     'https://raw.githubusercontent.com/allenai/allennlp'
# #     '/master/tutorials/tagger/validation.txt'))
# train_dataset = reader.read('../data/only_NER/train.txt')
# validation_dataset = reader.read('../data/only_NER/validation.txt')
# vocab = Vocabulary.from_instances(train_dataset + validation_dataset)
# EMBEDDING_DIM = 10
# HIDDEN_DIM =32
# token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
#                             embedding_dim=EMBEDDING_DIM)
# word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
# lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))
# model = LstmTagger(word_embeddings, lstm, vocab)
# optimizer = optim.SGD(model.parameters(), lr=0.1)
# iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
# iterator.index_with(vocab)
# trainer = Trainer(model=model,
#                   optimizer=optimizer,
#                   iterator=iterator,
#                   train_dataset=train_dataset,
#                   validation_dataset=validation_dataset,
#                   patience=10,
#                   num_epochs=1000)
# trainer.train()
# predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
# tag_logits = predictor.predict("There is a black square touching the edge.")['tag_logits']
# tag_ids = np.argmax(tag_logits, axis=-1)
# print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
# # Here's how to save the model.
# with open("../model_store/model.th", 'wb') as f:
#     torch.save(model.state_dict(), f)
# vocab.save_to_files("../model_store/vocabulary")
#
# # And here's how to reload the model.
# vocab2 = Vocabulary.from_files("../model_store/vocabulary")
# model2 = LstmTagger(word_embeddings, lstm, vocab2)
#
#
# with open("../model_store/model.th", 'rb') as f:
#     model2.load_state_dict(torch.load(f))
# predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
#
# train_read_file = open('../data/only_sentence/raw_train.json', 'r')
# train_write_file = open('../data/only_sentence/ner_train.json', 'w')
# for line in train_read_file:
#     tag_logits2 = predictor2.predict(line)['tag_logits']
#     tag_ids = np.argmax(tag_logits2, axis=-1)
#     res = [model.vocab.get_token_from_index(i, 'labels') for i in tag_ids]
#     for i in range(len(res)):
#         train_write_file.write(res[i]+' ')
#     # train_write_file.write(str(tag_logits2))
#     train_write_file.write('\n')
#     train_write_file.flush()
# train_read_file.close()
# train_write_file.close()
# print('finish')
