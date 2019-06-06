import os
s = os.sep
root = './'

entity_list = []

for rt, dirs, files in os.walk(root):
    for file in files:
        if file.endswith('.ann'):
            rd = open(str(file), 'r')
            generate_line = ''
            for line in rd:
                line = line.replace('\n', '').split('\t')
                if line[0].startswith('T'):
                    # BIO operation

                    # tr, lm and in operation
                    if line[1].startswith('spatial-entity'):
                        entity_list.append(line[2])

entity_set = set(entity_list)
print(entity_set)