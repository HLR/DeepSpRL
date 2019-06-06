from typing import Iterator, List, Dict
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn.functional import softmax

import sys
sys.path.append('../')
from kb.ontology import TOPOLOGY, DIRECTION


'''
generate the triplets and masks: no batch 
raw sentence:
    a circle touch a triangle and a triangle above a square

16: emb,    3 :token  5: triplet
  1. circle touch  triangle   (3, 16)
  2. triangle above  square   (3, 16)
  3. other, other, other      (3, 16)  <- mask
  4. other, other, other      (3, 16)
  5. other, other, other      (3, 16)
  (5, 3, 16)

'''

'''
type: no batch
  direction( left……..none)   topology: (none, ec….)
  1. left(cirle tou  triang)    (7, )  EC(cirle tou  triang)   (4, )
  2. right(tri above  square)   (7, )  DC(tri above  square)   (4, )
  3. none(other, other, other)  (7, )  none(other, other, other)(4, )
  4.  none(other, other, other) (7, )  none(other, other, other)(4, )
  5. none(other, other, other)  (7, )  none(other, other, other)(4, )
  (5 ,7)softmax1    (5,4)softmax2
concat()  -> (5, 10 ) 

'''
class Property_generating(Model):
    def __init__(self,
                 embed_size: int,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 ) -> None:
        super().__init__(vocab)
        self.embed_size = embed_size
        self.word_embeddings = word_embeddings
        self.test = torch.nn.Linear(in_features=embed_size,
                                                out_features=1)

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                structures1: Dict[str, torch.Tensor],
                structures2: Dict[str, torch.Tensor],
                structures3: Dict[str, torch.Tensor],
                ner: Dict[str, torch.Tensor],
                labels: torch.Tensor = None,
                ) -> Dict[str, torch.Tensor]:
        print('---------------')
        print(sentence)
        print(ner)
        # print(raw_sentence)
        # print(ner)
        print('---------------')

        '''(batch, 3, embed_size)'''
        # embeddings = self.word_embeddings(sentence)

        '''(batch, 3*embed_size)'''
        # batch_size, words_size, embed_size = embeddings.size()
        # embeddings_flat = embeddings.view(-1, self.embed_size)

        '''(batch, 4)'''
        # res = self.test(embeddings_flat)


    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     return {"accuracy": self.accuracy.get_metric(reset)}
