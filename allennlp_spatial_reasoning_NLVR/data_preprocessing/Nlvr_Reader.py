from typing import Iterator, List, Dict, Tuple
from allennlp.common.util import JsonDict

import json
import torch
import torch.optim as optim
import numpy as np
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, ListField, Field, ArrayField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

torch.manual_seed(1)

class NLVR_DatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    ''' read json file '''
    def read_json_line(self, line: str) -> Tuple[str, str, List[JsonDict], str, str, str, str]:
        data = json.loads(line)
        instance_id = data["identifier"]
        sentence = data["sentence"].replace('.', '').replace(',', '').replace('\n', '').split(' ')

        structured_reps = data["structured_rep"]
#         .replace('.', '')
        structures1 = self.describe_st_re(structured_reps, 0).split(' ')
        structures2 = self.describe_st_re(structured_reps, 1).split(' ')
        structures3 = self.describe_st_re(structured_reps, 2).split(' ')

        # label_strings = data["label"].lower()
        label_strings = '1' if data["label"].lower()=='true' else '0'
        return instance_id, sentence, structured_reps, label_strings, structures1, structures2, structures3

    def read_ner_line(self, file_name: str) -> List[str]:
        ner = []
        f = open(file_name, 'r')
        for line in f:
            ner.append(line.replace('\n',''))
        return ner

    def text_to_instance(self, ner: List[Token], tokens: List[Token], structures: List[JsonDict],
                         tag: str, max_length: int,
                         structures1: List[Token], structures2: List[Token], structures3: List[Token] ) -> Instance:


        sentence_field = TextField(tokens, self.token_indexers)

        # sentence_field = TextField([Token(x) for x in tokens], token_indexers=self.token_indexers)
        # print(text_field)
        sentence_len = len(tokens)
        if sentence_len >= max_length:
            mask = [1] * (max_length)
        else:
            mask = [1] * (sentence_len) + [0] * (max_length - sentence_len)

        # fields: Dict[str, Field] = {"sentence": sentence_field}
        fields = {"sentence": sentence_field}
        # fields['mask'] = ListField(mask)
        # fields['mask'] = ArrayField(np.array(mask))
        # fields['structures'] = ListField(structures)
        # fields['structures'] = Field(structures)
        '''using 11 dimension vector'''
        # structures1 = self.eleven_vector(structures, 0)
        # structures2 = self.eleven_vector(structures, 1)
        # structures3 = self.eleven_vector(structures, 2)

        '''using text structure: a small blue circle in (90, 20) with TextField'''
        fields['structures1'] = TextField(structures1, self.token_indexers)
        fields['structures2'] = TextField(structures2, self.token_indexers)
        fields['structures3'] = TextField(structures3, self.token_indexers)

        '''using text structure: a small blue circle in (90, 20) without TextField'''
        # fields['structures1'] = structures1
        # fields['structures2'] = structures2
        # fields['structures3'] = structures3

        fields["labels"] = LabelField(tag)

        # fields['raw_sentence'] = raw_sentence
        fields['ner'] = TextField(ner, self.token_indexers)
        return Instance(fields)

    def _read(self, file_path: List, max_length: int = 24) -> Iterator[Instance]:
        json_path = file_path[0]
        ner_path = file_path[1]

        ner = self.read_ner_line(ner_path)
        i=0
        with open(json_path) as f:
            for line in f:
                instance_id, sentence, structured_reps, label_strings, structures1, structures2, structures3 = self.read_json_line(line)
                yield self.text_to_instance(
                                            [Token(word) for word in ner[i].strip().split(' ')],
                                            [Token(word) for word in sentence],
                                            structured_reps, label_strings, max_length,
                                            [Token(word) for word in structures1],
                                            [Token(word) for word in structures2],
                                            [Token(word) for word in structures3],)
                i += 1

    ''' data processing for structure representation '''

    def describe_size_color(self, size, color):
        res1 = 'large' if size == 30 else 'middle' if size == 20 else 'small'
        res2 = 'yellow' if color == 'Yellow' else 'black' if color == 'Black' else 'blue'
        return res1 + ' ' + res2

    def describe_st_re(self, st_res, j):
        res = ''
        # information in each sub-image
        for k in range(len(st_res[j])):  # x
#             res += 'A ' + self.describe_size_color(st_res[j][k]['size'], st_res[j][k]['color']) + ' ' + st_res[j][k]['type'] + ' ' + 'in (' + str(st_res[j][k]['x_loc']) + ',' + str(st_res[j][k]['y_loc']) + ') '
            res += 'A ' + self.describe_size_color(st_res[j][k]['size'], st_res[j][k]['color']) + ' ' + st_res[j][k]['type'] + ' ' + 'in ' + str(st_res[j][k]['x_loc']) + ' and ' + str(st_res[j][k]['y_loc']) + ' . '
        return res

    def eleven_vector(self, st_res, j):
        feature_each_image = []
        # information in each sub-image
        for k in range(len(st_res[j])):  # x
            feature_each_obj = []
            feature_each_obj.append(st_res[j][k]['x_loc'])
            feature_each_obj.append(st_res[j][k]['y_loc'])
            if st_res[j][k]['size'] == 30:
                feature_each_obj.append(1)
                feature_each_obj.append(0)
                feature_each_obj.append(0)
            elif st_res[j][k]['size'] == 20:
                feature_each_obj.append(0)
                feature_each_obj.append(1)
                feature_each_obj.append(0)
            else:
                feature_each_obj.append(0)
                feature_each_obj.append(0)
                feature_each_obj.append(1)

            if st_res[j][k]['type'] == 'triangle':
                feature_each_obj.append(1)
                feature_each_obj.append(0)
                feature_each_obj.append(0)
            elif st_res[j][k]['type'] == 'square':
                feature_each_obj.append(0)
                feature_each_obj.append(1)
                feature_each_obj.append(0)
            else:
                feature_each_obj.append(0)
                feature_each_obj.append(0)
                feature_each_obj.append(1)

            if st_res[j][k]['color'] == 'Yellow':
                feature_each_obj.append(1)
                feature_each_obj.append(0)
                feature_each_obj.append(0)
            elif st_res[j][k]['color'] == 'Black':
                feature_each_obj.append(0)
                feature_each_obj.append(1)
                feature_each_obj.append(0)
            else:
                feature_each_obj.append(0)
                feature_each_obj.append(0)
                feature_each_obj.append(1)

            feature_each_image.append(feature_each_obj)

        for i in range(8 - len(st_res[j])):
           feature_each_image.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return feature_each_image

# reader = NLVR_DatasetReader()
# dev_dataset = reader.read('../chen/data/dev/dev.json')
# print(dev_dataset)


