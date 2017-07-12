__author__ = "David Ruddell"
# contact: dr1236@uncw.edu, dlruddell@gmail.com

"""
File to convert string values to integers for regression and other
analysis methods.
"""

import csv

old_data = open('headers3mgperml.csv', 'r')

new_data = open('massaged_data_test.csv', 'w')

csv_reader = csv.reader(old_data, delimiter=',')

csv_writer = csv.writer(new_data, delimiter=',')

for row in csv_reader:
    n_count = 0


    for n in row:
        if n == 'B73' or n == 'NORMAL' or n == 'CONTROL':
            row[n_count] = 0
        elif n == 'CML103' or n == 'LOW' or n == 'HIGH' or n == 'PAC':
            row[n_count] = 1
        elif n == 'PACGA':
            row[n_count] = 2
        elif n == 'UCN':
            row[n_count] = 3
        elif n == 'PCZ':
            row[n_count] = 4
        elif n == 'GA':
            row[n_count] = 5
        n_count += 1
    csv_writer.writerow(row)

