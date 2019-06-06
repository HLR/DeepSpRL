'''
Author: Chen Zheng
Goal: Splitting 1000 lines of one files to 1000 files, and seperately generate 1000 .txt files and
      1000 .ann files.
Date: 04/01/2019
'''


li = []
thousands_lines = open('raw_train_1000.txt', 'r')
for line in thousands_lines:
    li.append(line.replace('\n', ''))
thousands_lines.close()


for i in range(1000):
    file = open('train_'+str(i) + '.txt', 'w')
    file.write(li[i])
    file.close()
    file1 = open('train_'+str(i) + '.ann', 'w')
    file1.close()
