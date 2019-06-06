from typing import Iterator, List, Dict
import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy
from torch.nn.functional import softmax,sigmoid

import sys
sys.path.append('../')
from kb.ontology import TOPOLOGY, DIRECTION
from kb.entity_and_relation import *
from torch.autograd import Variable
from model.KB_checking import kb_checking
from kb.checking_property import entity_set_def
import torch.nn.functional as F


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
class All_generating(Model):
    def __init__(self,
                 embed_size: int,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 num_of_candidates: int,
                 ) -> None:
        super().__init__(vocab)
        self.embed_size = embed_size
        self.vocab = vocab
        self.topology_out_feature = len(TOPOLOGY)
        self.direction_out_feature = len(DIRECTION)
        self.word_embeddings = word_embeddings
        self.num_of_candidates = num_of_candidates
        # self.topology_predict = torch.nn.Linear(in_features=3*embed_size,
        #                                   out_features=self.topology_out_feature)
        # self.direction_predict = torch.nn.Linear(in_features=3 * embed_size,
        #                                   out_features=self.direction_out_feature)


        # TOPOLOGY = ['NONE', 'DC', 'EC', 'PP']
        # DIRECTION = ['ABOVE', 'BELOW', 'LEFT', 'RIGHT', 'NONE']

        self.Topology_None = torch.nn.Linear(in_features=3 * embed_size, out_features=2)
        self.Topology_DC = torch.nn.Linear(in_features=3 * embed_size, out_features=2)
        self.Topology_EC = torch.nn.Linear(in_features=3 * embed_size, out_features=2)
        self.Topology_PP = torch.nn.Linear(in_features=3 * embed_size, out_features=2)

        self.Dir_ABOVE = torch.nn.Linear(in_features=3 * embed_size, out_features=2)
        self.Dir_BELOW = torch.nn.Linear(in_features=3 * embed_size, out_features=2)
        self.Dir_LEFT = torch.nn.Linear(in_features=3 * embed_size, out_features=2)
        self.Dir_RIGHT = torch.nn.Linear(in_features=3 * embed_size, out_features=2)
        self.Dir_NONE = torch.nn.Linear(in_features=3 * embed_size, out_features=2)


        self.Topology_None = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )
        self.Topology_DC = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )
        self.Topology_EC = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )
        self.Topology_PP = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )
        self.Dir_ABOVE = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )
        self.Dir_BELOW = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )
        self.Dir_LEFT = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )
        self.Dir_RIGHT = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )
        self.Dir_NONE = torch.nn.Sequential(torch.nn.Linear(3 * embed_size, 2 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(2 * embed_size, 1 * embed_size),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(1 * embed_size, 2),
                                                 )


        self.accuracy = CategoricalAccuracy()

        self.loss_function = torch.nn.CrossEntropyLoss()
        # self.loss_function = torch.nn.MSELoss()
        self.kb_model = kb_checking()

    def forward(self,
                sentence: Dict[str, torch.Tensor],
                structures1: Dict[str, torch.Tensor],
                structures2: Dict[str, torch.Tensor],
                structures3: Dict[str, torch.Tensor],
                ner: Dict[str, torch.Tensor],
                labels: torch.Tensor = None,) -> Dict[str, torch.Tensor]:

        batch_size, _ = sentence['tokens'].size()

        '''generate pair'''
        pairs = self.generating_pair(sentence, ner)
        # print(pairs)
        property_checking_result = self.kb_model.checking_properity(structures1, structures2, structures3, pairs, self.vocab, batch_size)
        # print(property_checking_result)
        '''generate triple'''
        triplets = self.generating_triplt(sentence, ner)
        # print(triplets)

        '''(batch, candidata_triplets_number, 3, embed_size)'''
        embeddings = self.word_embeddings(triplets)

        # _, batch_size, _, _ = embeddings.size()

        '''(batch, candidata_triplets_number, 3*embed_size)'''
        embeddings_flat = embeddings.view(batch_size, self.num_of_candidates, 3*self.embed_size)
        # print(embeddings_flat.size())
        # embeddings_flat = embeddings_flat.permute(1, 0, 2)
        # print(embeddings_flat.size())

        # '''(candidata_triplets_number, batch, 9)'''
        # all_candiate_predict_res = torch.zeros(self.num_of_candidates, batch_size, 9)

        '''(batch, candidata_triplets_number, 9)'''
        all_candiate_predict_res = torch.zeros(batch_size, self.num_of_candidates, 9, 2)

        for i in range(batch_size):
            for j in range(self.num_of_candidates):

                # TOPOLOGY = ['NONE', 'DC', 'EC', 'PP']
                # DIRECTION = ['ABOVE', 'BELOW', 'LEFT', 'RIGHT', 'NONE']

                '''(batch, 1, 2)'''
                topo_none = self.Topology_None(embeddings_flat[i][j]).view(-1, 2)
                topo_dc = self.Topology_DC(embeddings_flat[i][j]).view(-1, 2)
                topo_ec = self.Topology_EC(embeddings_flat[i][j]).view(-1, 2)
                topo_pp = self.Topology_PP(embeddings_flat[i][j]).view(-1, 2)

                '''(batch, 1, 2)'''
                dir_above = self.Dir_ABOVE(embeddings_flat[i][j]).view(-1, 2)
                dir_below = self.Dir_BELOW(embeddings_flat[i][j]).view(-1, 2)
                dir_left = self.Dir_LEFT(embeddings_flat[i][j]).view(-1, 2)
                dir_right = self.Dir_RIGHT(embeddings_flat[i][j]).view(-1, 2)
                dir_none = self.Dir_NONE(embeddings_flat[i][j]).view(-1, 2)

                '''softmax'''
                topo_none = softmax(topo_none)
                topo_dc = softmax(topo_dc)
                topo_ec = softmax(topo_ec)
                topo_pp = softmax(topo_pp)

                dir_above = softmax(dir_above)
                dir_below = softmax(dir_below)
                dir_left = softmax(dir_left)
                dir_right = softmax(dir_right)
                dir_none = softmax(dir_none)

                '''sigmoid'''
                # topo_none = sigmoid(topo_none)
                # topo_dc = sigmoid(topo_dc)
                # topo_ec = sigmoid(topo_ec)
                # topo_pp = sigmoid(topo_pp)
                #
                # dir_above = sigmoid(dir_above)
                # dir_below = sigmoid(dir_below)
                # dir_left = sigmoid(dir_left)
                # dir_right = sigmoid(dir_right)
                # dir_none = sigmoid(dir_none)


                '''(batch, 9, 2)'''
                # 9 [1, 2]  --->   [9, 2]
                single_res = torch.cat([topo_none, topo_dc, topo_ec, topo_pp, dir_above, dir_below, dir_left, dir_right, dir_none], dim=0)

                all_candiate_predict_res[i][j] = single_res

        # ([5, 7, 9, 2])
        nlp_res = all_candiate_predict_res

        '''
        begin to check all triplets type prediction by distance 
        '''
        kb_res1,  kb_res2, kb_res3 = self.kb_model.checking_triplet_relation(structures1, structures2,
                        structures3, triplets, self.vocab, batch_size, self.num_of_candidates)


        predict_res1 = self.generate_final_true_or_false(property_checking_result, nlp_res, kb_res1, batch_size,
                                                         self.num_of_candidates)
        predict_res2 = self.generate_final_true_or_false(property_checking_result, nlp_res, kb_res2, batch_size,
                                                         self.num_of_candidates)
        predict_res3 = self.generate_final_true_or_false(property_checking_result, nlp_res, kb_res3, batch_size,
                                                         self.num_of_candidates)

        predict_res = self.integrate_res(predict_res1, predict_res2, predict_res3, batch_size, self.num_of_candidates)
        self.accuracy(torch.FloatTensor(predict_res).view(batch_size, 2), labels)


        output = {"logits": nlp_res}
        # print(nlp_res.size(), kb_res.size())

        gold_label = self.integrate_loss(kb_res1, kb_res2, kb_res3, batch_size, self.num_of_candidates)
        output["loss"] = self.loss_function(nlp_res.view(-1,2), gold_label.view(-1))
        # output["loss"] = self.loss_function(torch.FloatTensor(predict_res).view(batch_size, 2), labels)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

    '''generate all the candiate triplet'''
    def generating_triplt(self, sentence: Dict[str, torch.Tensor],
                          ner: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # print(self.vocab.get_token_from_index(9))
        # print(self.vocab.get_token_index('EM'))

        raw_sentence_list, ner_list = self.index_numpy_2_token_numpy(sentence, ner)
        raw_sentence_list, ner_list = self.split_list(raw_sentence_list, ner_list)

        entity_res, indicator_res = self.generate_entity_and_indicator(raw_sentence_list, ner_list)
        # print(entity_res)
        # print(indicator_res)

        '''
        In the above we have generate the 
        candidate entity ['tower', 'blocks'], and candidate indicator ['with']
        then combine these two list, to become [en1, indicator1, en2]....which is c x y
        we can have a upper bound of these candiate triplet, eg. 10.
        if <10, then mask some none, none, none
           >=10: pass because 10 candidate is already enough
        and put the word into the index, and feed into the next step's model. good luck!
        '''

        candiate_triplets = []
        for i in range(len(entity_res)):
            tmp_triplet, cur_entity, cur_indicator = [], entity_res[i], indicator_res[i]
            ent_pointer, ind_pointer = 0, 0
            while ind_pointer <= len(cur_indicator):

                if len(cur_indicator) == 0:
                    break
                if ent_pointer != len(cur_entity)-1 and len(cur_entity) > len(cur_indicator):
                    # print(cur_entity, ent_pointer, cur_indicator, ind_pointer)
                    # self.vocab.get_token_index('EM')
                    # tmp_triplet.append([cur_entity[ent_pointer], cur_indicator[ind_pointer], cur_entity[ent_pointer+1]])
                    tmp_triplet.append(
                        [self.vocab.get_token_index(cur_entity[ent_pointer]),
                         self.vocab.get_token_index(cur_indicator[ind_pointer]),
                         self.vocab.get_token_index(cur_entity[ent_pointer + 1]),
                         ]
                    )
                ent_pointer += 1
                ind_pointer += 1

            if len(tmp_triplet)<=self.num_of_candidates:
                for i in range(self.num_of_candidates-len(tmp_triplet)):
                    # tmp_triplet.append(['@@PADDING@@','@@PADDING@@','@@PADDING@@'])
                    tmp_triplet.append(
                    [self.vocab.get_token_index('@@PADDING@@'),
                     self.vocab.get_token_index('@@PADDING@@'),
                     self.vocab.get_token_index('@@PADDING@@'),
                     ]
                    )
            else:
                tmp_triplet = tmp_triplet[0:self.num_of_candidates]

            candiate_triplets.append(tmp_triplet)

        # print(candiate_triplets)
        tensor_candiate_triplets = torch.LongTensor(candiate_triplets)
        Res = {'tokens': tensor_candiate_triplets}
        # print(tensor_candiate_triplets)
        return Res


    '''generate all the property pairs'''
    def generating_pair(self, sentence: Dict[str, torch.Tensor],
                          ner: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raw_sentence_list, ner_list = self.index_numpy_2_token_numpy(sentence, ner)
        raw_sentence_list, ner_list = self.split_list(raw_sentence_list, ner_list)
        # print(raw_sentence_list, ner_list)
        res = self.check_property(raw_sentence_list)
        return res

    def index_numpy_2_token_numpy(self, sentence: Dict[str, torch.Tensor],
                          ner: Dict[str, torch.Tensor]):
        '''get raw sentence and ner list from  allennlp Instance'''
        raw_sentence_list = []
        ner_list = []

        '''get raw sentence index --> get raw sentence tokens'''
        for k, v in sentence.items():
            index_list = Variable(v).data.numpy()
            for i in range(len(index_list)):
                token_list = []
                for j in range(len(index_list[i])):
                    token_list.append(self.vocab.get_token_from_index(index_list[i][j]))
                raw_sentence_list.append(token_list)

        '''get ner index --> get ner tokens'''
        for k, v in ner.items():
            index_list = Variable(v).data.numpy()
            for i in range(len(index_list)):
                token_list = []
                for j in range(len(index_list[i])):
                    token_list.append(self.vocab.get_token_from_index(index_list[i][j]))
                ner_list.append(token_list)

        return raw_sentence_list, ner_list


    '''
    input: there is a box contain 2 triangle:
    output [there, is, a, box], 'contain', [2, triangle]
    [],'',[] or [],'',[],'',[]...
    '', str: indicator(only one word)
    []:entity, have space entity and property
    '''
    def split_list(self, raw_sentence, ner):
        new_sentence_list=[]
        new_ner_list=[]
        for batch_i in range(len(ner)):
            start = 0
            raw_sentence_split_list = []
            ner_split_list = []
            for list_j in range(len(ner[batch_i])):
                if ner[batch_i][list_j] == 'IN':
                    raw_sentence_split_list.append(raw_sentence[batch_i][start: list_j])
                    raw_sentence_split_list.append(raw_sentence[batch_i][list_j])
                    ner_split_list.append(ner[batch_i][start: list_j])
                    ner_split_list.append(ner[batch_i][list_j])
                    start = list_j + 1


                if ner[batch_i][list_j] == '@@PADDING@@':
                    raw_sentence_split_list.append(raw_sentence[batch_i][start: list_j])
                    ner_split_list.append(ner[batch_i][start: list_j])
                    break
                if list_j == len(ner[batch_i])-1:
                    raw_sentence_split_list.append(raw_sentence[batch_i][start: list_j+1])
                    ner_split_list.append(ner[batch_i][start: list_j+1])
                    break

            new_sentence_list.append(raw_sentence_split_list)
            new_ner_list.append(ner_split_list)

        return new_sentence_list, new_ner_list


    '''
    [   [    [],'',[]   ]  ,  [    [],'',[]   ], ...   ]
    first [ : sentence list               i
    second [ : the ith sentence           j
    []''[] : entity1, indicator, entity2  word_index
    '''
    def check_property(self, sentence_list):
        res = []
        # print('----------------------------')
        # print(sentence_list)
        for i in range(len(sentence_list)):
            property_each_sentence=[]
            for j in range(len(sentence_list[i])):
                if type(sentence_list[i][j]) == list:
                    sp_entity = 'None'
                    for word_index in range(len(sentence_list[i][j])):
                        if sentence_list[i][j][word_index] in entity_kb():
                            sp_entity = sentence_list[i][j][word_index]

                    for word_index in range(len(sentence_list[i][j])):
                        # or sentence_list[i][j][word_index] in shape_kb() \
                        if sentence_list[i][j][word_index] in number_kb() \
                                or sentence_list[i][j][word_index] in color_kb() \
                                or sentence_list[i][j][word_index] in size_kb():

                               #  or sentence_list[i][j][word_index] in spacial_kb() \
                               # or sentence_list[i][j][word_index] in logic_kb() \
                               # or sentence_list[i][j][word_index] in equal_kb() \
                                # or sentence_list[i][j][word_index] in compare_kb()\

                            # count = len(sentence_list[i][j]) - 1
                            # if sentence_list[i][j][count] == '@@PADDING@@':
                            #     count -= 1
                            # find the head_entity
                            # sp_entity = sentence_list[i][j][count]
                            property_each_sentence.append([sentence_list[i][j][word_index], sp_entity])

            res.append(property_each_sentence)
        return res

    '''
    [['box', 'top'], ['towers', 'block', 'top'], ['box', 'one', 'wall'], ['tower', 'blocks'], ['box', 'squares', 'middle']]
    [['with'], ['with', 'at'], ['with', 'touching'], ['with'], ['with', 'in']]
    '''
    def generate_entity_and_indicator(self, sentence_list, ner_list):
        entity_res = []
        indicator_res = []

        for i in range(len(ner_list)):
            tmp_indicator_res, tmp_entity_res = [], []
            for j in range(len(ner_list[i])):
                if 'IN' in ner_list[i]:

                    if ner_list[i][j] == 'IN':
                        tmp_indicator_res.append(sentence_list[i][j])
                        # print('indicator: ' + sentence_list[i][j])
                    else:
                        ''' chen 5.2.2019 add if condition, because may be the list is []-->empty'''
                        if len(sentence_list[i][j]) != 0:
                            count = len(sentence_list[i][j]) - 1
                            # if sentence_list[i][j][count] == '@@PADDING@@':
                            #     count -= 1

                            sp_entity = sentence_list[i][j][count]

                            tmp_entity_res.append(sp_entity)

            entity_res.append(tmp_entity_res)
            indicator_res.append(tmp_indicator_res)

        return entity_res, indicator_res


    def generate_final_true_or_false(self, pair_res, triplet_res, structure_res, batch_size, num_of_candidates):
        res = []
        for i in range(batch_size):
            # if pair_res[i] == 0:
            #     res.append([0.0, 1.0])
            # else:
            #     # is_break = 0
            #     # for j in range(num_of_candidates):
            #     #     for k in range(9):
            #     #         if triplet_res[i][j][k][1] > 0.5 and structure_res[i][j][k] == 0:
            #     #             res.append([0.0, 1.0])
            #     #             is_break = 1
            #     #             break
            #     #     if is_break == 1:
            #     #         break
            #     # if is_break == 0:
            #     #     res.append([1.0, 0.0])
            #     res.append([1.0, 0.0])

            if pair_res[i] == 0:
                res.append([0.0, 1.0])
            else:
                is_correct = 1
                for j in range(num_of_candidates):
                    err_count = 0
                    for k in range(9):
                        if triplet_res[i][j][k][1] > triplet_res[i][j][k][0] and structure_res[i][j][k] == 0:
                            err_count += 1
                    if err_count >= 2:
                        res.append([0.0, 1.0])
                        is_correct = 0
                        break
                    else:
                        continue
                if is_correct == 1:
                    res.append([1.0, 0.0])

        return res


    def integrate_res(self, predict_res1, predict_res2, predict_res3, batch_size, num_of_candidates):
        res = []
        for i in range(batch_size):
            if predict_res1[i][1] == 1.0 or predict_res2[i][1] == 1.0 or predict_res3[i][1] == 1.0:
                res.append([0.0, 1.0])
            else:
                res.append([1.0, 0.0])
        return res

    def integrate_loss(self, kb_res1, kb_res2, kb_res3, batch_size, num_of_candidates):
        res = []
        for i in range(batch_size):
            tmp_batch = []
            for j in range(num_of_candidates):
                tmp = []
                for k in range(9):
                    if kb_res1[i][j][k] == 1 or kb_res2[i][j][k] == 1 or kb_res3[i][j][k] == 1:
                        tmp.append(1)
                    else:
                        tmp.append(0)
                tmp_batch.append(tmp)
            res.append(tmp_batch)
            if tmp_batch[1] == 1 or tmp_batch[2] == 1:
                print(tmp_batch)
        return torch.LongTensor(res)

