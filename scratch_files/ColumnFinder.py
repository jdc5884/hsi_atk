# author: David Ruddell
# contact: dr1236@uncw.edu, dlruddell@gmail.com

import csv

with open("../Data/headers3mgperml.csv", 'r') as csvfile:
    line = csvfile.readline()
    line = line.split(',')
    count = 0
    print(line)
    for i in line[15:]:
        if float(i) > 644:
            print(count)
        count += 1

# Return 164. out of visible spectrum starts around 164 and forward.