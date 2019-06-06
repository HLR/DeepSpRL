import os
s = os.sep
root = './'

wr = open('700_ner_files.txt', 'w')

for rt, dirs, files in os.walk(root):
    for file in files:
        if file.endswith('.ann'):
            rd = open(str(file), 'r')
            generate_line = ''
            for line in rd:
                line = line.split('\t')
                # print(line)

                if line[1][0:2] in ['EO', 'EB', 'EI', 'EE', 'IN']:
                    generate_line += line[2].replace('\n','') + '###' + line[1][0:2] + ' '

                elif line[1][0:2] =='E ':
                    generate_line += line[2].replace('\n', '') + '###' + line[1][0] + ' '

            wr.write(generate_line[0:-1] + '\n')
            wr.flush()

            rd.close()
wr.close()