from __future__ import print_function, division
import numpy
import csv
import sys
from torch.utils.data import Dataset
import torch
import os
from sklearn.model_selection import StratifiedKFold
from select_features import generate_ge_file3


class CNN_Data(Dataset):
    """
    csv files ./lookuptxt/*.csv contains MRI filenames along with demographic and diagnosis information
    """

    def __init__(self, data_dir, categories, stage, cross_index, checkpoint_dir):
        self.data, self.labels = [], []
        self.cc, self.params_fit = None, None
        self.data_dir_mri = data_dir + "MRI/"
        subjects = []
        genes = []
        labels = []
        label_map = {categories[i]: i for i in range(len(categories))}
        with open(data_dir + "SNP/data_snp_v6.csv") as file:
            f_reader = list(csv.reader(file))
            for i, row in enumerate(f_reader):
                if i <= 1 or row[1] not in categories:
                    continue
                subjects.append(row[0])
                labels.append(label_map[row[1]])

        ind_all = numpy.array([i for i in range(len(subjects))])
        labels = numpy.array(labels).astype(numpy.int16)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        cv_splits = list(skf.split(ind_all, labels))
        tv_ind = ind_all[cv_splits[cross_index][0]]
        test_ind = ind_all[cv_splits[cross_index][1]]
        skf_tv = StratifiedKFold(n_splits=9, shuffle=True, random_state=cross_index)
        cv_splits_tv = list(skf_tv.split(tv_ind, labels[tv_ind]))
        train_ind = tv_ind[cv_splits_tv[0][0]]
        valid_ind = tv_ind[cv_splits_tv[0][1]]

        data_path_ge = "{}/GE_{}_{}.csv".format(checkpoint_dir, categories[0], categories[1])
        if stage == "train":
            generate_ge_file3(categories, data_dir + "GE/data_ge_v2.csv", data_path_ge,
                              is_extra=(True if 'CN' not in categories else False))

        data = [[] for _ in range(len(subjects))]
        data_ge = dict()
        data_ge_cn = [[] for _ in range(9)]
        with open("{}/GE_{}_{}.csv".format(checkpoint_dir, categories[0], categories[1]), 'r') as csv_file:
            f_reader = list(csv.reader(csv_file))
            for i, row in enumerate(f_reader):
                if i == 0:
                    genes = row[2:]
                    continue
                data_ge[row[0]] = list(map(float, row[2:]))
                if row[1] == 'CN':
                    for j, value in enumerate(row[2:]):
                        data_ge_cn[j].append(float(row[j + 2]))
        self.cc, self.params_fit = self.linfit(data_ge_cn)
        self.cc, self.params_fit = self.cc.cuda(), self.params_fit.cuda()
        data_snp = dict()
        with open(data_dir + "SNP/data_snp_v6.csv", 'r') as file:
            f_reader = list(csv.reader(file))
            for i, row in enumerate(f_reader):
                if i <= 1 or row[1] not in categories:
                    continue
                data_snp[row[0]] = []
                gene2data = dict()
                for j, value in enumerate(row):
                    if j <= 1:
                        continue
                    if f_reader[0][j] not in gene2data:
                        gene2data[f_reader[0][j]] = []
                    gene2data[f_reader[0][j]].append(int(value))
                for gene in genes:
                    data_snp[row[0]] = data_snp[row[0]] + (gene2data[gene] +
                                                           [0 for _ in range(337 - len(gene2data[gene]))])

        for i, subject in enumerate(subjects):
            ge = torch.tensor(data_ge[subject]).cuda()
            ge = torch.unsqueeze(ge, dim=1)
            ge_pre = torch.mul(self.params_fit[0], ge) + self.params_fit[1]
            ge_div = self.cc * (ge_pre - torch.squeeze(ge, dim=1))
            ge_div = ge_div.reshape(81)
            data[i] = ge_div.cpu().tolist() + data_snp[subject]

        data = torch.tensor(data).to(torch.float32).cuda()
        labels = torch.tensor(labels).to(torch.int64).cuda()
        subjects = numpy.array(subjects)

        if stage == "test":
            self.data = data[test_ind]
            self.labels = labels[test_ind]
            self.subjects = subjects[test_ind]
        elif stage == "train":
            self.data = data[train_ind]
            self.labels = labels[train_ind]
            self.subjects = subjects[train_ind]
        else:
            self.data = data[valid_ind]
            self.labels = labels[valid_ind]
            self.subjects = subjects[valid_ind]

        self.micro_all = data[train_ind]
        self.label_all = labels[train_ind]

    def get_class_weight(self):
        npos = 0
        nneg = 0
        for i, label in enumerate(self.labels):
            if label == 0:
                npos += 1
            else:
                nneg += 1
        return [npos / (npos + nneg), nneg / (npos + nneg)]

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def linfit(data_ge_cn):
        data_ge_cn = numpy.array(data_ge_cn)
        cc = torch.zeros((data_ge_cn.shape[0], data_ge_cn.shape[0]))
        params_fit = torch.zeros((2, cc.shape[0], cc.shape[1]))
        for i in range(9):
            params_fit[0][i][i], params_fit[1][i][i] = 1, 0
            for j in range(i + 1, 9):
                cc_ = numpy.corrcoef(data_ge_cn[i], data_ge_cn[j])[0][1]
                if -0.2 <= cc_ <= 0.2:
                    continue
                cc[i][j], cc[j][i] = abs(cc_), abs(cc_)
                param = numpy.polyfit(data_ge_cn[i], data_ge_cn[j], deg=1)
                params_fit[0][i][j], params_fit[1][i][j] = param[0], param[1]
                params_fit[0][j][i], params_fit[1][j][i] = 1 / param[0], -param[1]
        return cc, params_fit

    def __getitem__(self, idx):
        label = self.labels[idx]
        ge_snp = self.data[idx]
        mri = numpy.load(self.data_dir_mri + self.subjects[idx] + ".npy").reshape(182 * 218 * 182)
        mri = torch.from_numpy(mri).to(torch.float32).cuda()
        idx = torch.tensor([idx]).cuda()
        data = torch.cat((idx, ge_snp, mri), dim=0)
        data = torch.unsqueeze(data, dim=0)
        return data, label, self.micro_all, self.label_all


class Ex_Data(Dataset):
    """
    csv files ./lookuptxt/*.csv contains MRI filenames along with demographic and diagnosis information
    """

    def __init__(self, data_name, data_dir):
        self.data, self.labels = [], []
        self.cc, self.params_fit = None, None
        self.data_dir_mri = data_dir + "MRI/"
        subjects = []
        genes = []
        labels = []
        name = data_name
        group2label = {'CN': 0, 'MCI': 1}
        with open('./' + name + ".csv") as file:
            f_reader = list(csv.reader(file))
            for i, row in enumerate(f_reader):
                if i < 1:
                    continue
                subjects.append(row[0])
                labels.append(group2label[row[1]])

        data_ge_cn = [[] for _ in range(9)]
        with open("./data/support/ge.csv", 'r') as csv_file:
            f_reader = list(csv.reader(csv_file))
            for i, row in enumerate(f_reader):
                if i == 0:
                    continue
                if row[1] == 'CN':
                    for j, value in enumerate(row[2:]):
                        data_ge_cn[j].append(float(row[j + 2]))
        self.cc, self.params_fit = self.linfit(data_ge_cn)
        self.cc, self.params_fit = self.cc.cuda(), self.params_fit.cuda()

        data_ge = dict()
        with open('./data/' + name + '/ge.csv', 'r') as file:
            f_reader = list(csv.reader(file))
            for i, row in enumerate(f_reader):
                if len(row) != 10:
                    print('Incorrect gene expression data for subject {}!'.format(row[0]))
                    sys.exit()
                if i < 1:
                    genes = row[1:]
                    continue
                if row[0] not in subjects:
                    continue
                data_ge[row[0]] = list(map(float, row[1:]))
        data = [[] for _ in range(len(subjects))]
        data_snp = dict()
        with open('./data/' + name + '/snp.csv', 'r') as file:
            f_reader = list(csv.reader(file))
            for i, row in enumerate(f_reader):
                if i < 2 or row[0] not in subjects:
                    continue
                data_snp[row[0]] = []
                gene2data = dict()
                for j, value in enumerate(row):
                    if j < 1:
                        continue
                    if f_reader[0][j] not in gene2data:
                        gene2data[f_reader[0][j]] = []
                    gene2data[f_reader[0][j]].append(int(value))
                for gene in genes:
                    if len(gene2data[gene]) > 337:
                        print('Incorrect snp data for subject {}!'.format(row[0]))
                        sys.exit()
                    data_snp[row[0]] = data_snp[row[0]] + (gene2data[gene] +
                                                           [0 for _ in range(337 - len(gene2data[gene]))])

        for i, subject in enumerate(subjects):
            if subject not in data_ge:
                print('Gene expression data for subject {} does not exist!'.format(subject))
            if subject not in data_snp:
                print('SNP data for subject {} does not exist!'.format(subject))
            if subject + '.npy' not in os.listdir(self.data_dir_mri):
                print('MRI data for subject {} does not exist!'.format(subject))
            ge = torch.tensor(data_ge[subject]).cuda()
            ge = torch.unsqueeze(ge, dim=1)
            ge_pre = torch.mul(self.params_fit[0], ge) + self.params_fit[1]
            ge_div = self.cc * (ge_pre - torch.squeeze(ge, dim=1))
            ge_div = ge_div.reshape(81)
            data[i] = ge_div.cpu().tolist() + data_snp[subject]

        data = torch.tensor(data).to(torch.float32).cuda()
        self.subjects = numpy.array(subjects)
        self.data = data
        self.micro_all = self.get_support()
        # labels = [0] * 20
        self.labels = torch.tensor(labels).to(torch.int64).cuda()

    def get_support(self):
        subject_all = []
        micro_all = []
        data_ge = dict()
        with open('./data/support/ge.csv', 'r') as file:
            f_reader = list(csv.reader(file))
            for i, row in enumerate(f_reader):
                if i < 1:
                    genes = row[2:]
                    continue
                subject_all.append(row[0])
                data_ge[row[0]] = list(map(float, row[2:]))
        data_snp = dict()
        with open('./data/support/snp.csv', 'r') as file:
            f_reader = list(csv.reader(file))
            for i, row in enumerate(f_reader):
                if i < 2:
                    continue
                data_snp[row[0]] = []
                gene2data = dict()
                for j, value in enumerate(row):
                    if j < 2:
                        continue
                    if f_reader[0][j] not in gene2data:
                        gene2data[f_reader[0][j]] = []
                    gene2data[f_reader[0][j]].append(int(value))
                for gene in genes:
                    data_snp[row[0]] = data_snp[row[0]] + (gene2data[gene] +
                                                           [0 for _ in range(337 - len(gene2data[gene]))])
        for s, subject in enumerate(subject_all):
            ge = torch.tensor(data_ge[subject]).cuda()
            ge = torch.unsqueeze(ge, dim=1)
            ge_pre = torch.mul(self.params_fit[0], ge) + self.params_fit[1]
            ge_div = self.cc * (ge_pre - torch.squeeze(ge, dim=1))
            ge_div = ge_div.reshape(81)
            micro_all.append(ge_div.cpu().tolist() + data_snp[subject])
        return torch.tensor(micro_all).cuda()

    def __len__(self):
        return self.data.shape[0]

    @staticmethod
    def linfit(data_ge_cn):
        data_ge_cn = numpy.array(data_ge_cn)
        cc = torch.zeros((data_ge_cn.shape[0], data_ge_cn.shape[0]))
        params_fit = torch.zeros((2, cc.shape[0], cc.shape[1]))
        for i in range(9):
            params_fit[0][i][i], params_fit[1][i][i] = 1, 0
            for j in range(i + 1, 9):
                cc_ = numpy.corrcoef(data_ge_cn[i], data_ge_cn[j])[0][1]
                if -0.2 <= cc_ <= 0.2:
                    continue
                param = numpy.polyfit(data_ge_cn[i], data_ge_cn[j], deg=1)
                params_fit[0][i][j], params_fit[1][i][j] = param[0], param[1]
                params_fit[0][j][i], params_fit[1][j][i] = 1 / param[0], -param[1]
        return cc, params_fit

    def __getitem__(self, idx):
        ge_snp = self.data[idx]
        mri = numpy.load(self.data_dir_mri + self.subjects[idx] + ".npy")
        s = 1
        for i in range(len(mri.shape)):
            s = s * mri.shape[i]
        if s != 182 * 218 * 182:
            print('Incorrect mri data for subject {}!'.format(self.subjects[idx]))
            sys.exit()
        mri = mri.reshape(182 * 218 * 182)
        mri = torch.from_numpy(mri).to(torch.float32).cuda()
        idx = torch.tensor([idx]).cuda()
        data = torch.cat((idx, ge_snp, mri), dim=0)
        data = torch.unsqueeze(data, dim=0)
        label = self.labels[idx]
        return data, self.micro_all, label
