from torch.autograd import Variable
import kb.entity_and_relation
import kb.checking_property as checking_property
import kb.checking_triplet_type as checking_type
import kb.entity_and_relation as er
from kb.ontology import TOPOLOGY, DIRECTION
import torch

class kb_checking():
    def __init__(self):
        super().__init__()

    def checking_properity(self, image_structure1, image_structure2, image_structure3,  pairs, vocab, batch_size):
        res1 = self.check_each_structure_property(image_structure1, pairs, vocab, batch_size)
        res2 = self.check_each_structure_property(image_structure2, pairs, vocab, batch_size)
        res3 = self.check_each_structure_property(image_structure3, pairs, vocab, batch_size)
        res = [sum(x) for x in zip(res1, res2, res3)]
        for i in range(len(res)):
            if res[i] >= 1:
                res[i] = 1
        return res


    def check_each_structure_property(self, image_structure1, pairs, vocab, batch_size):
        pair_T_or_false = [1] * batch_size
        # sentence_image_structure = []
        for i in range(len(image_structure1['tokens'])):
            # print(i)
            current_image_structure = Variable(image_structure1['tokens'][i]).data.numpy()
            current_image_structure_word = []
            for j in range(len(current_image_structure)):
                if vocab.get_token_from_index(current_image_structure[j]) != '@@PADDING@@':
                    current_image_structure_word.append(vocab.get_token_from_index(current_image_structure[j]))
            '''
            structure to sentence word list
            '''
            final_processing_structure = []
            current_image_structure_word.remove(current_image_structure_word[-1])
            for j in range(0, len(current_image_structure_word), 9):
                final_processing_structure.append(current_image_structure_word[j:j + 9])

            '''
            Checking that whether all of the pairs are showing in the structure representation
            '''
            # print('----------->',pairs)
            # pairs:  (batch size, each pair list, each pair)  <----3 dimension

            # how to check? Related to the knowledge base
            ## pairs[i] = [['a', 'box'], ['a', 'item'], ['black', 'item'], ['same', 'item'], ['color', 'item'], ['and', 'item'], ['no', 'item']]
            # pairs[i][k] = ['a', 'box']<-pairs[i][0]

            # print('------------->', pairs[i])
            for k in range(len(pairs[i])):
                # print('-----<<<', pairs[i][k])
                if pairs[i][k][1] == 'box' or pairs[i][k][1] == 'boxes':
                    continue
                # checking it is have a tower and checking kb that it is have a tower
                if (pairs[i][k][1] == 'tower'):
                    tmp_res = checking_property.tower_checking(pairs[i][k], final_processing_structure)
                    if tmp_res is False:
                        pair_T_or_false[i] = 0
                        # print('tower')
                        break
                if (pairs[i][k][0] in er.color_kb() and pairs[i][k][1] != 'tower'):
                    tmp_res = checking_property.color_checking(pairs[i][k], final_processing_structure)
                    if tmp_res is False:
                        pair_T_or_false[i] = 0
                        # print('color')
                        break
                if (pairs[i][k][0] in er.number_kb() and pairs[i][k][1] != 'tower'):
                    tmp_res = checking_property.quantity_checking(pairs[i][k], final_processing_structure)
                    if tmp_res is False:
                        pair_T_or_false[i] = 0
                        # print('quantity')
                        break

                        # sentence_image_structure.append(final_processing_structure)

        # print(sentence_image_structure)
        return pair_T_or_false




    def checking_triplet_relation(self, image_structure1, image_structure2, image_structure3,  triplets, vocab, batch_size, number_of_candiates):
        res1 = self.check_each_structure_type(image_structure1, triplets, vocab, batch_size, number_of_candiates)
        res2 = self.check_each_structure_type(image_structure2, triplets, vocab, batch_size, number_of_candiates)
        res3 = self.check_each_structure_type(image_structure3, triplets, vocab, batch_size, number_of_candiates)
        # print(res1)
        # print(res2)
        # print(res3)
        # res = [sum(x) for x in zip(res1, res2, res3)]
        # for i in range(len(res)):
        #     if res[i] >= 1:
        #         res[i] = 1
        res1 = torch.LongTensor(res1)
        res2 = torch.LongTensor(res2)
        res3 = torch.LongTensor(res3)

        return res1, res2, res3



    def check_each_structure_type(self, image_structure1, triplets, vocab, batch_size, number_of_candiates):

        triplets_numpy = Variable(triplets['tokens']).data.numpy()
        triplets_words = []
        for i in range(batch_size):
            tmp_candid = []
            for j in range(number_of_candiates):
                tmp_list = []
                tmp_list.append(vocab.get_token_from_index(int(triplets_numpy[i][j][0])))
                tmp_list.append(vocab.get_token_from_index(int(triplets_numpy[i][j][1])))
                tmp_list.append(vocab.get_token_from_index(int(triplets_numpy[i][j][2])))

                tmp_candid.append(tmp_list)
            triplets_words.append(tmp_candid)

        # (batch, number_of_candiates, number of topology + number of direction)
        # all_triplet_type_res = [[[0]*(len(TOPOLOGY)+len(DIRECTION))] * number_of_candiates]*batch_size
        all_triplet_type_res = []
        # sentence_image_structure = []
        for i in range(len(image_structure1['tokens'])):
            # print(i)
            current_image_structure = Variable(image_structure1['tokens'][i]).data.numpy()
            current_image_structure_word = []
            for j in range(len(current_image_structure)):
                if vocab.get_token_from_index(current_image_structure[j]) != '@@PADDING@@':
                    current_image_structure_word.append(vocab.get_token_from_index(current_image_structure[j]))
            '''
            structure to sentence word list
            '''
            final_processing_structure = []
            current_image_structure_word.remove(current_image_structure_word[-1])
            for j in range(0, len(current_image_structure_word), 9):
                final_processing_structure.append(current_image_structure_word[j:j + 9])
            # print(final_processing_structure)

            '''
            Checking that all of the triplets belong to which type
            '''
            # print(triplets_numpy)
            # tmp_type_res : (number of candidates, len(TOPOLOGH)+len(DIRECTION))
            # TOPOLOGY = ['NONE', 'DC', 'EC', 'PP']
            # DIRECTION = ['ABOVE', 'BELOW', 'LEFT', 'RIGHT', 'NONE']
            tmp_type_res = []
            for k in range(len(triplets_words[i])):
                # print(triplets_words[i][k])

                # box with sth
                if(triplets_words[i][k][0] == 'box' or triplets_words[i][k][0] == 'boxes'
                   or triplets_words[i][k][0] == 'tower' or triplets_words[i][k][0] == 'towers') \
                        and triplets_words[i][k][1] == 'with':
                    # tmp_type_res.append([0, 0, 0, 1, 0, 0, 0, 0])
                    tmp_type_res.append([0, 0, 0, 1, 0, 0, 0, 0, 1])

                    # print([0, 0, 0, 1, 0, 0, 0, 0, 1])
                elif triplets_words[i][k][0] == '@@PADDING@@' or triplets_words[i][k][2] == '@@PADDING@@' :
                    tmp_type_res.append([1, 0, 0, 0, 0, 0, 0, 0, 1])
                    # tmp_type_res.append([1, 0, 0, 0, 0, 0, 0, 0])
                elif checking_property.remove_s_es(triplets_words[i][k][0]) in er.shape_kb() \
                    and checking_property.remove_s_es(triplets_words[i][k][2]) in er.shape_kb():
                    # tmp_type_res.append([1, 0, 0, 0, 0, 0, 0, 0, 1])
                    x1, y1, x2, y2, size1, size2 = checking_type.check_x_and_y_location_for_shape_in_kb(
                        checking_property.remove_s_es(triplets_words[i][k][0]),
                        checking_property.remove_s_es(triplets_words[i][k][2]),
                        final_processing_structure,
                    )
                    res = checking_type.generate_type_res(x1, y1, x2, y2, size1, size2)
                    tmp_type_res.append(res)
                    # print(res)

                else:
                    tmp_type_res.append([1, 0, 0, 0, 0, 0, 0, 0, 1])
                    # tmp_type_res.append([1, 0, 0, 0, 0, 0, 0, 0])

            all_triplet_type_res.append(tmp_type_res)


        return all_triplet_type_res

