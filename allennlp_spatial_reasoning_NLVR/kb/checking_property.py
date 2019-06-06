import kb.entity_and_relation as er

'''
[[['a', 'block'], ['yellow', 'block'], ['a', 'block'], ['yellow', 'block']], 
[['a', 'tower'], ['exactly', 'blocks'], ['three', 'blocks'], ['a', 'block'], ['yellow', 'block']], 
[['a', 'block'], ['black', 'block'], ['a', 'tower'], ['three', 'blocks']], 
[['a', 'tower'], ['exactly', 'blocks'], ['two', 'blocks'], ['a', 'block'], ['blue', 'block']], 
[['a', 'block'], ['yellow', 'block'], ['above', 'block'], ['a', 'block'], ['black', 'block']]]

[
 [['A', 'middle', 'blue', 'square', 'in', '40', 'and', '80', '.'], 
  ['A', 'middle', 'yellow', 'square', 'in', '40', 'and', '59', '.']], 

 [['A', 'middle', 'blue', 'square', 'in', '40', 'and', '80', '.'], 
  ['A', 'middle', 'blue', 'square', 'in', '40', 'and', '59', '.'], 
  ['A', 'middle', 'black', 'square', 'in', '40', 'and', '38', '.'], 
  ['A', 'middle', 'blue', 'square', 'in', '40', 'and', '17', '.']], 
  
 [['A', 'middle', 'blue', 'square', 'in', '40', 'and', '80', '.'], 
  ['A', 'middle', 'blue', 'square', 'in', '40', 'and', '59', '.'], 
  ['A', 'middle', 'blue', 'square', 'in', '40', 'and', '38', '.'], 
  ['A', 'middle', 'black', 'square', 'in', '40', 'and', '17', '.']], 
  
  [['A', 'middle', 'black', 'square', 'in', '40', 'and', '80', '.']], 
  
  [['A', 'middle', 'yellow', 'square', 'in', '40', 'and', '80', '.'], 
   ['A', 'middle', 'blue', 'square', 'in', '40', 'and', '59', '.'], 
   ['A', 'middle', 'black', 'square', 'in', '40', 'and', '38', '.'], 
   ['A', 'middle', 'black', 'square', 'in', '40', 'and', '17', '.']]
]

'''


'''
block also means one of triangle, circle and square.
'''

def remove_s_es(word):
    if word.endswith('ies'):
        return word[0:len(word)-3]+'y'
    if word.endswith('s'):
        return word[0: len(word)-1]
    return word

# checking quantity
def quantity_checking(pair, structure):
    # print(pair)
    entity = remove_s_es(pair[1])
    str_arr = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    int_arr = [1,2,3,4,5,6,7,8,9]
    ''' just check the box is contain this object '''
    if pair[0] == 'a' or pair[0] == 'an':
        pair[0] = 1
    if pair[0] == 'some':
        pair[0] = 2
    if pair[0] == 'all':
        pair[0] = len(structure)
    if pair[0] in str_arr:
        pair[0] = int_arr[str_arr.index(pair[0])]

    '''for tri, square, square'''
    if entity not in er.general_entity_kb():
        count = 0
        for i in range(len(structure)):
            if entity == structure[i][3]:
                count += 1
                if count == pair[0]:
                    # print(count, pair[0], entity, structure[i][3])
                    return True
    # for item, obj, etc
    else:
        # for tmp_entity in er.shape_kb():
        #     count = 0
        #     entity = tmp_entity
        #     for i in range(len(structure)):
        #         if entity == structure[i][3]:
        #             count += 1
        #             if count == pair[0]:
        #                 print(count, pair[0], entity)
        #                 return True
        if pair[0] <= len(structure):
            # print(pair[0], len(structure), entity)
            return True
    return False

# checking color
def color_checking(pair, structure):
    # print(pair)
    entity = remove_s_es(pair[1])
    '''as long as exist this color, then true'''
    '''for tri, square, square'''
    if entity not in er.general_entity_kb():
        for i in range(len(structure)):
            if entity == structure[i][3] and pair[0] == structure[i][2]:
                # print(pair[0], structure[i][2], entity, structure[i][3])
                return True
    # for item, obj, etc
    else:
        for i in range(len(structure)):
            if pair[0] == structure[i][2]:
                # print(pair[0], structure[i][2])
                return True
    return False


# checking spatial
def spatial_checking(pair, structure):
    entity = remove_s_es(pair[1])


# checking whether kb have a tower or not
def tower_checking(pair, structure):
    # print(pair)
    # print(structure)
    # dict = {'small': 10, 'middle': 20, 'large': 30}
    for i in range(len(structure)-1):
        for j in range(i+1, len(structure)):
            # ['A', 'middle', 'yellow', 'square', 'in', '40', 'and', '80', '.']
            if int(structure[i][5]) == int(structure[j][5]) and \
                structure[i][3] == 'square' and structure[j][3] == 'square':
                # (int(structure[i][7]) + dict[structure[i][1]] == int(structure[j][7]) or \
                #  int(structure[i][7]) - dict[structure[i][1]] == int(structure[j][7])):
                return True
    return False

def entity_set_def():
    return ['one', 'top', 'box', 'ones', 'side', 'triangles', 'other', 'circle', 'block', 'triangle', 'towers', 'square', 'edge', 'item', 'line', 'squares', 'blocks', 'objects', 'items', 'together', 'corner', 'each', 'tower', 'circles', 'circle', 'base', 'it', 'its', 'black', 'objetcs', 'object', 'boxes', 'wall']

