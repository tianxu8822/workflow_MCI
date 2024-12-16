import os

import numpy

from model import Model
from utils import matrix_sum, get_acc, get_MCC, get_confusion_matrix, write_raw_score
from dataloader import CNN_Data, Ex_Data
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import csv
from sklearn.metrics import accuracy_score


class Experiment_wrapper:
    def __init__(self,
                 fil_num,
                 drop_rate,
                 seed,
                 batch_size,
                 data,
                 data_dir,
                 categories,
                 learn_rate,
                 train_epoch,
                 gpu=0,
                 metric='ACC'):

        """
            :param fil_num:    output channel number of the first convolution layer
            :param drop_rate:  dropout rate of the last 2 layers, see model.py for details
            :param seed:       random seed
            :param batch_size: batch size for training CNN
            :param metric:     metric used for saving model during training, can be either 'accuracy' or 'MCC'
                               for example, if metric == 'accuracy', then the time point where validation set has best
                               accuracy will be saved
        """

        self.epoch = 0
        self.seed = seed
        self.categories = categories
        self.data_name = data
        self.Data_dir = data_dir
        self.learn_rate = learn_rate
        self.train_epoch = train_epoch
        self.batch_size = batch_size
        self.cross_index = None
        self.eval_metric = get_acc if metric == 'ACC' else get_MCC
        self.checkpoint_dir = None
        self.model = Model(fil_num=fil_num, drop_rate=drop_rate).cuda()
        self.optimal_epoch = self.epoch
        self.optimal_valid_metric = 0.0
        self.optimal_valid_matrix = [[0, 0], [0, 0], [0, 0]]
        self.flag = True

        self.train_dataloader, self.valid_dataloader, self.test_dataloader = None, None, None
        self.optimizer = None
        self.criterion_clf = nn.CrossEntropyLoss().cuda()
        self.gpu = gpu
        self.skip_epoch = 0

    def cross_validation(self, cross_index):
        self.cross_index = cross_index
        self.checkpoint_dir = './checkpoint_dir/{}_{}/exp{}/'.format(self.categories[0], self.categories[1],
                                                                     self.cross_index)
        self.prepare_dataloader()
        # self.train()
        # self.test()

    def external_verification(self):
        print('external verification ... ')
        test_data = Ex_Data(self.data_name, self.Data_dir)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.model.load_state_dict(torch.load('data/support/model.pth', map_location={'cuda:0': 'cuda:{}'.format(self.gpu)}))
        self.model.train(False)
        result = dict()
        with torch.no_grad():
            dataloader = self.test_dataloader
            prob_all, label_all = [], []
            for batch, (inputs, micro_all, labels) in enumerate(dataloader):
                micro_all = micro_all[0]
                clf_output, output_micro, output_mri, _, _ = self.model(inputs, None, micro_all, None)
                clf_output = torch.softmax(clf_output, dim=1).cpu()
                prob_all = prob_all + clf_output.detach().tolist()
                label_all = label_all + labels.cpu().tolist()
                idxes = inputs[:, 0, 0].cpu()
                for i in range(idxes.shape[0]):
                    result[test_data.subjects[int(idxes[i])]] = [clf_output[i][0].item(), clf_output[i][1].item()]
        pred_all = numpy.argmax(numpy.array(prob_all), axis=1)
        acc = accuracy_score(label_all, pred_all)
        print('ACC: {:.4f}'.format(acc))
        with open('../result.csv', 'w', newline='\n') as csv_file:
            f_writer = csv.writer(csv_file)
            f_writer.writerow(['Subjectid', 'CN', 'MCI', 'Group'])
            for s, subject in enumerate(test_data.subjects):
                f_writer.writerow([subject, result[subject][0], result[subject][1],
                                  'CN' if result[subject][0] > result[subject][1] else 'MCI'])

    def prepare_dataloader(self):
        train_data = CNN_Data(self.Data_dir, stage='train', cross_index=self.cross_index, categories=self.categories,
                              checkpoint_dir=self.checkpoint_dir)
        valid_data = CNN_Data(self.Data_dir, stage='valid', cross_index=self.cross_index, categories=self.categories,
                              checkpoint_dir=self.checkpoint_dir)
        test_data = CNN_Data(self.Data_dir, stage='test', cross_index=self.cross_index, categories=self.categories,
                             checkpoint_dir=self.checkpoint_dir)
        class_weight = train_data.get_class_weight()
        loss_weight = 1 / (torch.tensor(class_weight).cuda() * len(class_weight))
        self.criterion_clf.weight = loss_weight
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_dataloader = DataLoader(valid_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def train(self):
        print("training ...")
        self.optimal_valid_matrix = [[0, 0], [0, 0], [0, 0]]
        self.optimal_epoch = 0
        dis_para = list(map(id, self.model.gcn_layer_1.parameters())) + list(map(id, self.model.gcn_layer_2.parameters()))
        base_para = filter(lambda p: id(p) not in dis_para, self.model.parameters())
        self.optimizer = optim.Adam([{'params': base_para},
                                     {'params': self.model.gcn_layer_1.parameters(), 'lr': 0.005},
                                     {'params': self.model.gcn_layer_2.parameters(), 'lr': 0.001}],
                                    lr=self.learn_rate, betas=(0.5, 0.999))
        for self.epoch in range(self.train_epoch):
            self.train_model_epoch()
            valid_matrix = self.valid_model_epoch('valid')
            print('{}th epoch validation confusion matrix:'.format(self.epoch), valid_matrix)
            print('eval_metric:', "%.4f" % self.eval_metric(valid_matrix))
            with open(self.checkpoint_dir + "valid_result.txt", 'a') as file:
                file.write(str(self.epoch) + ' ' + str(valid_matrix) + ' ' +
                           "%.6f" % self.eval_metric(valid_matrix) + ('\n' if self.epoch <= 99 else ' '))
            if self.epoch <= 99:
                continue
            self.save_checkpoint(valid_matrix)
            test_matrix = self.valid_model_epoch('test')
            with open(self.checkpoint_dir + "valid_result.txt", 'a') as file:
                file.write(str(test_matrix) + ' ' + "%.6f" % self.eval_metric(test_matrix) + '\n')
        print('Best model saved at the {}th epoch:'.format(self.optimal_epoch), self.optimal_valid_metric,
              self.optimal_valid_matrix)
        return self.optimal_valid_metric

    def train_model_epoch(self):
        self.model.train(True)
        loss_all = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).cuda()
        for batch, (inputs, labels, micro_all, label_all) in enumerate(self.train_dataloader):
            micro_all, label_all = micro_all[0], label_all[0]
            self.model.zero_grad()

            clf_output, output_micro, output_mri, co_loss, cons_loss = \
                self.model(inputs, labels, micro_all, label_all)

            clf_loss = self.criterion_clf(clf_output, labels)
            da_loss = self.domain_align(output_micro, output_mri, labels)
            micro_loss = self.criterion_clf(output_micro, labels)
            mri_loss = self.criterion_clf(output_mri, labels)

            loss_all[0] += clf_loss
            loss_all[1] += da_loss
            loss_all[2] += co_loss
            loss_all[3] += cons_loss
            loss_all[4] += micro_loss
            loss_all[5] += mri_loss

            loss = clf_loss + co_loss + cons_loss + micro_loss + mri_loss
            if self.epoch >= 50:
                loss += da_loss
            loss.backward()
            self.optimizer.step()

        loss_all /= len(self.train_dataloader)
        with open(self.checkpoint_dir + "train_loss.txt", 'a') as file:
            loss_all = loss_all.cpu().tolist()
            loss_all = ['{:.5}'.format(loss) for loss in loss_all]
            file.write(str(self.epoch) + ' ' + str(loss_all) + '\n')

    @staticmethod
    def domain_align(output_micro, output_mri, labels):
        da_loss = torch.tensor(0.0001).cuda()
        if output_micro.shape[0] != 4:
            return da_loss
        num = 0
        for i in range(labels.shape[0]):
            for j in range(i + 1, labels.shape[0]):
                if labels[i] != labels[j]:
                    num += 1
        if num == 0:
            return da_loss

        output_micro = torch.softmax(output_micro, dim=1)
        output_mri = torch.softmax(output_mri, dim=1)
        distance_micro = torch.zeros((num, 3)).cuda()
        distance_mri = torch.zeros((num, 3)).cuda()
        distance_label = torch.zeros((num, 3)).cuda()
        labels_oh = torch.zeros((labels.shape[0], 3)).cuda()
        labels_oh[(torch.arange(start=0, end=labels_oh.shape[0]), labels)] = 1
        idx = 0
        for i in range(output_micro.shape[0]):
            for j in range(i + 1, output_micro.shape[0]):
                if labels[i] == labels[j]:
                    continue
                distance_micro[idx] = (output_micro[i] - output_micro[j])
                distance_mri[idx] = (output_mri[i] - output_mri[j])
                distance_label[idx] = (labels_oh[i] - labels_oh[j])
                idx += 1

        for i in range(distance_label.shape[0]):
            new_loss = torch.pow(torch.exp(distance_micro[i] * distance_label[i]) -
                                 torch.exp(distance_mri[i] * distance_label[i]), 2)[0]
            da_loss = da_loss + new_loss
            if torch.isnan(new_loss):
                print(i, distance_mri, distance_micro, distance_label)
                print(output_micro)
                print(output_mri)
                print(labels, labels_oh)
                import sys
                sys.exit()
        return da_loss

    def valid_model_epoch(self, flag):
        dataloader = self.valid_dataloader if flag == "valid" else self.test_dataloader
        with torch.no_grad():
            self.model.train(False)
            matrix = [[0, 0], [0, 0], [0, 0]]
            for batch, (inputs, labels, micro_all, label_all) in enumerate(dataloader):
                micro_all, label_all = micro_all[0], label_all[0]
                clf_output, output_micro, output_mri, _, _ = self.model(inputs, labels, micro_all, label_all)

                matrix = matrix_sum(matrix, get_confusion_matrix(clf_output, labels))
        return matrix

    def save_checkpoint(self, valid_matrix):
        if self.flag or (self.eval_metric(valid_matrix) >= self.optimal_valid_metric - self.skip_epoch * 0.0002):
            self.optimal_epoch = self.epoch
            self.optimal_valid_matrix = valid_matrix
            self.optimal_valid_metric = self.eval_metric(valid_matrix)
            for root, Dir, Files in os.walk(self.checkpoint_dir):
                for File in Files:
                    if File.endswith('.pth'):
                        os.remove(os.path.join(self.checkpoint_dir, File))
            torch.save(self.model.state_dict(), '{}epoch_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch))
            self.flag = False
            self.skip_epoch = 0
        else:
            self.skip_epoch += 1

    def test(self):
        print('{} testing ... '.format(self.cross_index))

        for root, dirs, files in os.walk(self.checkpoint_dir):
            for file in files:
                if file[-4:] == '.pth':
                    self.optimal_epoch = file[6:]
                    self.optimal_epoch = int(self.optimal_epoch[:-4])

        self.model.load_state_dict(torch.load('{}epoch_{}.pth'.format(self.checkpoint_dir, self.optimal_epoch)))

        self.model.train(False)
        with torch.no_grad():
            for stage in ['train', "valid", 'test']:
                if stage == 'train':
                    dataloader = self.train_dataloader
                elif stage == "valid":
                    dataloader = self.valid_dataloader
                else:
                    dataloader = self.test_dataloader
                f_clf = open(self.checkpoint_dir + 'raw_score_clf_info_{}.txt'.format(stage), 'w')
                f_micro = open(self.checkpoint_dir + 'raw_score_micro_info_{}.txt'.format(stage), 'w')
                f_mri = open(self.checkpoint_dir + 'raw_score_mri_info_{}.txt'.format(stage), 'w')
                f_result = open(self.checkpoint_dir + 'cross_index_{}.txt'.format(stage), 'w')
                matrix_clf = [[0, 0], [0, 0], [0, 0]]
                matrix_micro = [[0, 0], [0, 0], [0, 0]]
                matrix_mri = [[0, 0], [0, 0], [0, 0]]
                for batch, (inputs, labels, micro_all, label_all) in enumerate(dataloader):
                    micro_all, label_all = micro_all[0], label_all[0]
                    clf_output, output_micro, output_mri, _, _ = self.model(inputs, labels, micro_all, label_all)

                    write_raw_score(f_clf, clf_output, labels)
                    write_raw_score(f_micro, output_micro, labels)
                    write_raw_score(f_mri, output_mri, labels)

                    matrix_clf = matrix_sum(matrix_clf, get_confusion_matrix(clf_output, labels))
                    matrix_micro = matrix_sum(matrix_micro, get_confusion_matrix(output_micro, labels))
                    matrix_mri = matrix_sum(matrix_mri, get_confusion_matrix(output_mri, labels))
                print(stage, 'confusion matrix: clf {}, micro {}, mri {}'.format(matrix_clf, matrix_micro, matrix_mri))
                print("Accuracy: clf {:.4f}, micro {:.4f}, mri {:.4f}".format(self.eval_metric(matrix_clf),
                                                                              self.eval_metric(matrix_micro),
                                                                              self.eval_metric(matrix_mri)))
                f_result.write(str(matrix_clf) + "\n" + str(matrix_micro) + '\n' + str(matrix_mri))
                f_clf.close()
                f_micro.close()
                f_mri.close()
                f_result.close()
