import kb.entity_and_relation as er

def some_entity_infor():
    # base: y == 0
    # edge and wall: x == 0 or x == 100 or y == 0 or y == 100
    # corner: (x == 0 and y == 0) or (x == 0 and y == 100 ) or (x == 100 and y == 0) or (x == 100 and y == 100)
    return 0


def check_x_and_y_location_for_shape_in_kb(obj1, obj2, image_struecture):
    x1, y1, x2, y2 = -1, -1, -1, -1
    size1 = -1,
    size2 = -1
    for i in range(len(image_struecture)):

        if obj1 in image_struecture[i]:
            x1 = int(image_struecture[i][5])
            y1 = int(image_struecture[i][7])
            if image_struecture[i][1] == 'small':
                size1 = 10
            elif image_struecture[i][1] == 'middle':
                size1 = 20
            if image_struecture[i][1] == 'large':
                size1 = 30
        if obj2 in image_struecture[i]:
            x2 = int(image_struecture[i][5])
            y2 = int(image_struecture[i][7])
            if image_struecture[i][1] == 'small':
                size2 = 10
            elif image_struecture[i][1] == 'middle':
                size2 = 20
            if image_struecture[i][1] == 'large':
                size2 = 30
    if size1 == (-1,):
        # print('come in')
        size1 = 10
    if size2 == -1:
        size2 = 10
    # print("lalala", size1, size2)
    return x1, y1, x2, y2, size1, size2

def generate_type_res(x1, y1, x2, y2, size1, size2):
    # print('come here!')
    result = []

    # NONE
    if x1 == None or x2 == None or y1 == None or y2 == None:
        result.append(1)
    else:
        result.append(0)

    # DC
    # print(size1, size2)
    if x1 + size1 < x2 -10 or x2 + size2 < x1 -10 or y1 + size1 < y2 - 10 or y2 + size2 < y1 - 10:
        result.append(1)
    else:
        result.append(0)


    # EC
    if (x1 + size1 <= x2+5 and x1 + size1 >= x2-10)\
        or (x2 + size2 <= x1+5 and x2 + size2 >= x1-10) \
        or (y1 + size1 <= y2 + 5 and y1 + size1 >= y2 - 10) \
        or (y2 + size2 <= y1 + 5 and y2 + size2 >= y1 - 10):
        result.append(1)
    else:
        result.append(0)

    # PP always 0 for obj1 in the shape list, and obj2 also in the shape list
    result.append(0)

    # ABOVE
    result.append(1) if er.is_above(x1, x2, y1, y2) is True else result.append(0)
    # BELOW
    result.append(1) if er.is_below(x1, x2, y1, y2) is True else result.append(0)
    # LEFT
    result.append(1) if er.is_left(x1, x2, y1, y2) is True else result.append(0)
    # RIGHT
    result.append(1) if er.is_right(x1, x2, y1, y2) is True else result.append(0)
    # None
    if x1 == None or x2 == None or y1 == None or y2 == None:
        result.append(1)
    else:
        result.append(0)

    # print(result)
    return result




def kb_compute_triplet_type(entity1, entity2, x1, y1, size1, x2, y2, size2):
    result = []
    TOPOLOGY = ['NONE', 'DC', 'EC', 'PP']
    DIRECTION = ['ABOVE', 'BELOW', 'LEFT', 'RIGHT', 'FRONT', 'BACK', 'NONE']

    # NONE
    if entity1 == None or entity2 == None or x1 == None or x2 == None or y1 == None or y2 == None:
        result.append([0])
    # DC
    if x1 + size1 != x2 and x2 + size2 != x2 and y1 + size1 != y2 and y2 + size2 != y2:
        result.append([1])
    # EC
    if x1 + size1 == x2 or x2 + size2 == x2 or y1 + size1 == y2 or y2 + size2 == y2:
        result.append([2])
    # PP
    if entity1 in er.pp_entity_kb() or entity2 in er.pp_entity_kb():
        result.append([3])
    ''' PP relation have no direction relations'''

    # ABOVE
    result.append([4]) if er.is_above(x1, x2, y1, y2) is True else None
    # BELOW
    result.append([5]) if er.is_below(x1, x2, y1, y2) is True else None
    # LEFT
    result.append([6]) if er.is_left(x1, x2, y1, y2) is True else None
    # RIGHT
    result.append([7]) if er.is_right(x1, x2, y1, y2) is True else None
    # front
    # back
    # None
    if entity1 in er.pp_entity_kb() or entity2 in er.pp_entity_kb() or result[0] == [3]:
        result.append([10])

    return result