import csv
import numpy
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE


def ttest(samples1, samples2):
    samples1 = numpy.array(samples1)
    samples2 = numpy.array(samples2)
    t, pvalue = stats.levene(samples1, samples2)
    if pvalue > 0.05:
        t, pvalue = stats.ttest_ind(samples1, samples2)
    else:
        t, pvalue = stats.ttest_ind(samples1, samples2, equal_var=False)
    return pvalue


def generate_ge_file(categories, subjects_train, srcpath, tarpath):
    with open(srcpath, 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
    pvalue2pos = dict()
    group2pos = dict()
    for category in categories:
        group2pos[category] = []
    for i, row in enumerate(f_reader):
        if i == 3:
            for j, group in enumerate(row):
                if j == 0 or group == '' or group not in group2pos:
                    continue
                group2pos[group].append(j)
        elif i >= 10:
            group2sample = dict()
            for category in categories:
                group2sample[category] = []
            for j, sample in enumerate(row):
                if j <= 2 or j == len(row) - 1 or f_reader[2][j] not in subjects_train \
                        or (j not in group2pos[categories[0]] and j not in group2pos[categories[1]]):
                    continue
                group2sample[f_reader[3][j]].append(float(sample))
            pvalue = ttest(group2sample[categories[0]], group2sample[categories[1]])
            if pvalue not in pvalue2pos:
                pvalue2pos[pvalue] = []
            pvalue2pos[pvalue].append(i)

    pvalues = sorted(pvalue2pos.keys())
    gene_poss = []
    i = 0
    while i <= 1023:
        gene_poss += pvalue2pos[pvalues[i]]
        if len(gene_poss) > 1024:
            gene_poss = gene_poss[:1024]
            break
        i += 1
    subject_poss = []
    for category in categories:
        subject_poss = subject_poss + group2pos[category]

    with open(tarpath, 'w', newline='') as csv_file:
        f_writer = csv.writer(csv_file)
        new_row = ["Subjectid", "Group"]
        for j in gene_poss:
            new_row.append(f_reader[j][2])
        f_writer.writerow(new_row)
        for subject_pos in subject_poss:
            new_row = [f_reader[2][subject_pos], f_reader[3][subject_pos]]
            for gene_pos in gene_poss:
                new_row.append(f_reader[gene_pos][subject_pos])
            f_writer.writerow(new_row)


def generate_ge_file2(categories, subjects_train, srcpath, tarpath):
    with open(srcpath, 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
    pos = set()
    data = []
    labels = []
    subject2label = dict()
    train_ind = []
    labelmap = {categories[i]: i for i in range(len(categories))}
    idx2gene = dict()
    for i, row in enumerate(f_reader):
        if i == 3:
            idx = 0
            for j, label in enumerate(row):
                if j == 0 or label == '' or label not in labelmap:
                    continue
                pos.add(j)
                labels.append(labelmap[label])
                data.append([])
                subject2label[f_reader[2][j]] = label
                if f_reader[2][j] in subjects_train:
                    train_ind.append(idx)
                idx += 1
        elif i >= 10:
            idx2gene[(len(data[0]))] = row[2]
            idx = 0
            for j, value in enumerate(row):
                if j <= 2 or j == len(row) - 1 or j not in pos:
                    continue
                data[idx].append(float(value))
                idx += 1
    data = numpy.array(data)
    labels = numpy.array(labels)
    train_ind = numpy.array(train_ind)

    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=1024, step=100, verbose=0)
    selector = selector.fit(data[train_ind], labels[train_ind].ravel())
    data = selector.transform(data)

    with open(tarpath, 'w', newline='') as csv_file:
        f_writer = csv.writer(csv_file)
        new_row = ["Subjectid", "Group"]
        for j in idx2gene.keys():
            if selector.support_[j]:
                new_row.append(idx2gene[j])
        f_writer.writerow(new_row)
        for i, subject in enumerate(subject2label.keys()):
            new_row = [subject, subject2label[subject]] + data[i].tolist()
            f_writer.writerow(new_row)


def generate_ge_file3(categories, srcpath, tarpath, is_extra):
    if is_extra:
        categories = categories + ['CN']
    labelsmap = {categories[i]: i for i in range(len(categories))}
    with open(srcpath, 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
        genes = ["ABCA7", "APOE", "BIN1", "CD2AP", "CD33", "CLU", "CR1", "MS4A6A", "PICALM"]
        p_values = [1 for _ in range(len(genes))]
        subjects, group, data, labels = [], [], [], []
        for i, row in enumerate(f_reader):
            if i == 3:
                for j, label in enumerate(row):
                    if label not in categories:
                        continue
                    subjects.append(f_reader[2][j])
                    group.append(f_reader[3][j])
                    data.append([0 for _ in range(len(genes))])
                    labels.append(labelsmap[f_reader[3][j]])
            elif i >= 10 and f_reader[i][2] in genes:
                gene_pos = genes.index(f_reader[i][2])
                idx = 0
                samples = [[], []]
                data_tmp = []
                for j, value in enumerate(row):
                    if f_reader[3][j] in categories:
                        data_tmp.append(float(f_reader[i][j]))
                        if is_extra and f_reader[3][j] == 'CN':
                            idx += 1
                            continue
                        samples[labels[idx]].append(float(f_reader[i][j]))
                        idx += 1
                p_value = ttest(samples[0], samples[1])
                if p_value < p_values[gene_pos]:
                    for j in range(len(data)):
                        data[j][gene_pos] = data_tmp[j]
    with open(tarpath, 'w', newline='') as csv_file:
        f_writer = csv.writer(csv_file)
        head = ["Subjectid", "Group"]
        for gene in genes:
            head.append(gene)
        f_writer.writerow(head)
        for i, subject in enumerate(subjects):
            new_row = [subject, group[i]] + data[i]
            f_writer.writerow(new_row)


def generate_snp_file(categories, subjects_train, srcpath, tarpath):
    with open(srcpath, 'r') as csv_file:
        f_reader = list(csv.reader(csv_file))
    pvalue2pos = dict()
    for i in range(len(f_reader[0])):
        if i <= 1:
            continue
        group2sample = dict()
        for category in categories:
            group2sample[category] = []
        for j in range(len(f_reader)):
            if j == 0 or f_reader[j][1] not in group2sample or f_reader[j][0] not in subjects_train:
                continue
            value = int(f_reader[j][i])
            if value != -1:
                group2sample[f_reader[j][1]].append(value)
        pvalue = ttest(group2sample[categories[0]], group2sample[categories[1]])
        if pvalue not in pvalue2pos:
            pvalue2pos[pvalue] = []
        pvalue2pos[pvalue].append(i)
    pvalues = sorted(pvalue2pos.keys())
    site_poss = []
    i = 0
    while i <= 255:
        site_poss = site_poss + pvalue2pos[pvalues[i]]
        if len(site_poss) > 256:
            site_poss = site_poss[:256]
            break
        i += 1
    with open(tarpath, 'w', newline='') as csv_file:
        f_writer = csv.writer(csv_file)
        for i, row in enumerate(f_reader):
            if i != 0 and f_reader[i][1] not in categories:
                continue
            new_row = []
            for j in range(len(f_reader[i])):
                if j <= 1 or j in site_poss:
                    new_row.append(f_reader[i][j])
            f_writer.writerow(new_row)
