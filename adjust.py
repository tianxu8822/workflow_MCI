import csv
import numpy
import torch
import os


def adjust():
    subject_all = []
    categories = ['CN', 'MCI']
    with open('data/ADNI/SNP/data_snp_v6.csv', 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
    content_snp = f_reader[:2]
    for i, row in enumerate(f_reader):
        if i < 2 or row[1] not in categories:
            continue
        content_snp.append(row)
        subject_all.append(row[0])
    with open('data/support/snp.csv', 'w', newline='\n') as csv_file:
        f_writer = csv.writer(csv_file)
        f_writer.writerows(content_snp)
    with open('checkpoint_dir/CN_MCI/exp0/GE_CN_MCI.csv', 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
    content_ge = f_reader
    with open('data/support/ge.csv', 'w', newline='\n') as csv_file:
        f_writer = csv.writer(csv_file)
        f_writer.writerows(content_ge)


def get_info():
    files = os.listdir('data/ADNI/MRI')
    subjects = [files[i][:-4] for i in range(len(files))]
    with open('data/example/snp.csv', 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
        content_snp = f_reader[:2]
        for i, row in enumerate(f_reader):
            if i < 2:
                continue
            if row[0] in subjects:
                content_snp.append(row)
        for i in range(len(content_snp)):
            del content_snp[i][1]
    with open('data/example/snp.csv', 'w', newline='\n') as csv_file:
        f_writer = csv.writer(csv_file)
        f_writer.writerows(content_snp)
    with open('data/example/ge.csv', 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
        content_ge = f_reader[:2]
        for i, row in enumerate(f_reader):
            if i < 2:
                continue
            if row[0] in subjects:
                content_ge.append(row)
        for i in range(len(content_ge)):
            del content_ge[i][1]
    with open('data/example/ge.csv', 'w', newline='\n') as csv_file:
        f_writer = csv.writer(csv_file)
        f_writer.writerows(content_ge)

    with open('example.csv', 'w', newline='\n') as csv_file:
        f_writer = csv.writer(csv_file)
        content = [[subjects[i]] for i in range(len(subjects))]
        f_writer.writerows(content)


def overlap():
    subject_all = []
    with open('data/support/snp.csv', 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
        for i, row in enumerate(f_reader):
            if i < 2:
                continue
            subject_all.append(row[0])
    content = []
    with open('data/support/ge.csv', 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
        for i, row in enumerate(f_reader):
            if i == 0 or row[0] in subject_all:
                content.append(row)
    with open('data/support/ge.csv', 'w', newline='\n') as csv_file:
        f_writer = csv.writer(csv_file)
        f_writer.writerows(content)


overlap()
