import os
import json

last_count = 1

def format_change(file1, file2, name):
    print(name)
    python2json = {}

    sentence = []
    word_index = []
    word_char_index = []
    ner = []

    tr = []
    lm = []
    indi = []

    match = {}

    triplet = []
    pair = []
    store_R = []
    store_E = []
    '''
    spatial-entity
    trajector
    landmark
    indicator
    quantity
    color-value
    shape-value
    size-value
    spa-value
    logic-value
    compare-value
    '''
    triplet_count = 0

    for line in file1:
        line = line.replace('\n', '').split('\t')
        if line[0].startswith('T'):
            # BIO operation
            if line[1].startswith('EB') or \
                    line[1].startswith('EM') or \
                    line[1].startswith('EE') or \
                    line[1].startswith('E') or \
                    line[1].startswith('IN') or \
                    line[1].startswith('EO'):

                sentence.append(line[2])
                tmp = line[1].split(' ')
                ner.append({'word_index': str(int(line[0][1:]) - 1), 'bio': tmp[0], 'word': line[2]})
                word_index.append(str(int(line[0][1:]) - 1))
                word_char_index.append(tmp[1])

    for line in file2:
        line = line.replace('\n', '').split('\t')
        if line[0].startswith('T'):
            # BIO operation

            # tr, lm and in operation
            if line[1].startswith('trajector'):
                print('come in')
                # tr[sentence.index(line[2])] = line[2]
                tr.append({"word_index": sentence.index(line[2]), 'trajector': line[2]})
            elif line[1].startswith('landmark'):
                # lm[sentence.index(line[2])] = line[2]
                lm.append({"word_index": sentence.index(line[2]), 'landmark': line[2]})
            elif line[1].startswith('indicator'):
                # indi[sentence.index(line[2])] = line[2]
                indi.append({"word_index": sentence.index(line[2]), 'indicator': line[2]})
            # pairs matching operation
            # if line[1].startswith('quantity') or \
            #     line[1].startswith('spatial-entity') or \
            #     line[1].startswith('color-value') or \
            #     line[1].startswith('shape-value') or \
            #     line[1].startswith('size-value') or \
            #     line[1].startswith('spa-value') or \
            #     line[1].startswith('logic-value') or \
            #     line[1].startswith('compare-value'):
            match[line[0]] = str(sentence.index(line[2]))

        # R1\tcount Arg1:T11 Arg2:T12\t\n
        elif line[0].startswith('R'):
            store_R.append(line)

        # 'E1\tPP:T21 arg1:T19 arg2:T18 arg3:T20\n'
        elif line[0].startswith('E'):
            store_E.append(line)

    # print(tr)
    for line in store_R:
        tmp = line[1].split(' ')
        property = str(match[tmp[1][5:]])
        head_entity = str(match[tmp[2][5:]])
        new_dict = {}
        new_dict[property] = tmp[0]
        # if head_entity in pair.keys():
        #     pair[head_entity].append(new_dict)
        # else:
        #     pair[head_entity] = []
        #     pair[head_entity].append(new_dict)
        '''change in jfk airport'''
        pair.append({'head_entity_index': head_entity, 'properity_index': property, 'properity_word': tmp[0]})

    for line in store_E:
        tmp = line[1].split(' ')
        if tmp[0].startswith('PP') or \
                tmp[0].startswith('EC') or \
                tmp[0].startswith('DC'):

            # triplet[triplet_count] = {}
            type = tmp[0].split(':')
            # triplet[triplet_count]['type'] = type[0]
            # triplet[triplet_count]['indicator'] = str(match[tmp[1][5:]])
            # triplet[triplet_count]['trajector'] = str(match[tmp[2][5:]])
            # triplet[triplet_count]['landmark'] =  str(match[tmp[3][5:]])
            # print('----------->tmp: '+str(len(tmp)))
            if len(tmp) == 4:
                triplet.append(
                    {'id': triplet_count, 'general_type': 'region', 'specific_type': 'Direction', 'type': type[0],
                     'indicator': str(match[tmp[1][5:]]),
                     'trajector': str(match[tmp[2][5:]]),
                     'landmark': str(match[tmp[3][5:]])})

                triplet_count += 1
        elif tmp[0].startswith('ABOVE') or \
                tmp[0].startswith('BELOW') or \
                tmp[0].startswith('LEFT') or \
                tmp[0].startswith('RIGHT'):

            type = tmp[0].split(':')
            if len(tmp) == 4:
                triplet.append(
                    {'id': triplet_count, 'general_type': 'region', 'specific_type': 'Direction', 'type': type[0],
                     'indicator': str(match[tmp[1][5:]]),
                     'trajector': str(match[tmp[2][5:]]),
                     'landmark': str(match[tmp[3][5:]])})

    # print(sentence)
    # print(word_index)
    # # print(word_char_index)
    # print(ner)
    # print(tr, lm, indi)
    # print(pair)
    # print(triplet)

    global last_count
    python2json["sid"] = 'sid'+str(last_count)
    last_count += 1
    python2json["sentence"] = sentence
    python2json["word_index"] = word_index
    python2json['ner'] = ner
    python2json['trajectors'] = tr
    python2json['landmarks'] = lm
    python2json['indicators'] = indi
    python2json['pairs'] = pair
    python2json['triplets'] = triplet

    json_str = json.dumps(python2json)


    return json_str





s = os.sep
root = '../700_multi_data/'

wr = open('brat_result.json', 'w')

for rt, dirs, files in os.walk(root):
    for file in files:
        if file.endswith('.ann'):
            rd1 = open(root + str(file), 'r')
            rd2 = open(root + str(file), 'r')
            res = format_change(rd1, rd2, file)
            # print(res)
            wr.write(res+'\n')
            wr.flush()

            rd1.close()
            rd2.close()
wr.close()
