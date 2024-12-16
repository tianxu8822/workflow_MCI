from __future__ import print_function, division

from abc import ABC
import torch.nn as nn
import torch


class ConvLayer(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, is_pooling, pooling=(1, 1, 0), relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.is_pooling = is_pooling
        if self.is_pooling:
            self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type == 'leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate) 
       
    def forward(self, x):
        x = self.conv(x)
        if self.is_pooling:
            x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class GCN(nn.Module, ABC):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, 1, bias=False),
        )

    def forward(self, micro, label, micro_all, label_all):
        dis = torch.zeros((micro.shape[0], micro_all.shape[0], micro_all.shape[1])).cuda()
        for i in range(micro.shape[0]):
            dis[i] = micro_all - micro[i]
        dis = torch.pow(dis, 2)
        anti_dis = torch.squeeze(self.fc(dis), dim=2) - torch.mean(dis, dim=2)
        anti_dis = torch.softmax(anti_dis, dim=1)
        anti_dis_argsort = torch.argsort(anti_dis, dim=1)

        # anti_dis_cut = torch.zeros_like(anti_dis).cuda()
        # for i in range(anti_dis.shape[0]):
        #     for j in range(anti_dis.shape[1]):
        #         if anti_dis[i][j] >= 2 / (micro_all.shape[0]):
        #             anti_dis_cut[i][j] = anti_dis[i][j]
        #         if j + 1 >= anti_dis.shape[1] * 0.99 and anti_dis_cut[i][anti_dis_argsort[i][j]] == 0:
        #             anti_dis_cut[i][anti_dis_argsort[i][j]] = anti_dis[i][anti_dis_argsort[i][j]]

        k = 6
        knn_mask = torch.zeros((micro.shape[0], micro_all.shape[0])).cuda()
        sort_idx_left = torch.arange(micro.shape[0]).repeat(k).view(k, micro.shape[0]).T
        knn_mask[(sort_idx_left, anti_dis_argsort[:, micro_all.shape[0] - k:])] = 1
        anti_dis_cut = anti_dis * knn_mask
        anti_dis_cut = anti_dis_cut / torch.sum(anti_dis_cut, dim=1, keepdim=True)
        micro_tmp = torch.zeros_like(micro).cuda()
        micro_tmp[:] = micro[:] + torch.matmul(anti_dis_cut[:], micro_all)
        if label is None or label_all is None:
            co_loss = None
        else:
            co_loss = self.composition_loss(label, label_all, anti_dis_cut)
        return micro_tmp, co_loss

    @staticmethod
    def composition_loss(label, label_all, anti_dis_cut):
        dis_label = torch.zeros_like(anti_dis_cut).cuda()
        for i in range(label.shape[0]):
            dis_label[i] = torch.abs(label_all - label[i])
        co_loss = torch.tensor(0.0001).cuda() + torch.matmul(dis_label.view(-1), anti_dis_cut.view(-1)) / label.shape[0]
        return co_loss


class Model(nn.Module, ABC):
    def __init__(self, drop_rate, fil_num):
        super(Model, self).__init__()
        self.conv_1 = ConvLayer(1, fil_num, drop_rate, (7, 2, 3), True, (3, 2, 1))

        self.block_1_1 = ConvLayer(fil_num, fil_num, drop_rate, (3, 1, 1), False)
        self.block_1_2 = ConvLayer(fil_num, fil_num, drop_rate, (3, 1, 1), False)

        self.block_3_1 = ConvLayer(fil_num, 2 * fil_num, drop_rate, (3, 2, 1), False)
        self.block_3_2 = ConvLayer(2 * fil_num, 2 * fil_num, drop_rate, (3, 1, 1), False)
        self.block_3_3 = ConvLayer(fil_num, 2 * fil_num, drop_rate, (1, 2, 0), False)

        self.block_4_1 = ConvLayer(2 * fil_num, 2 * fil_num, drop_rate, (3, 1, 1), False)
        self.block_4_2 = ConvLayer(2 * fil_num, 2 * fil_num, drop_rate, (3, 1, 1), False)

        self.block_5_1 = ConvLayer(2 * fil_num, 4 * fil_num, drop_rate, (3, 2, 1), False)
        self.block_5_2 = ConvLayer(4 * fil_num, 4 * fil_num, drop_rate, (3, 1, 1), False)
        self.block_5_3 = ConvLayer(2 * fil_num, 4 * fil_num, drop_rate, (1, 2, 0), False)

        self.block_6_1 = ConvLayer(4 * fil_num, 4 * fil_num, drop_rate, (3, 1, 1), False)
        self.block_6_2 = ConvLayer(4 * fil_num, 4 * fil_num, drop_rate, (3, 1, 1), False)

        self.block_7_1 = ConvLayer(4 * fil_num, 8 * fil_num, drop_rate, (3, 2, 1), False)
        self.block_7_2 = ConvLayer(8 * fil_num, 8 * fil_num, drop_rate, (3, 1, 1), False)
        self.block_7_3 = ConvLayer(4 * fil_num, 8 * fil_num, drop_rate, (1, 2, 0), False)

        self.block_8_1 = ConvLayer(8 * fil_num, 8 * fil_num, drop_rate, (3, 1, 1), False)
        self.block_8_2 = ConvLayer(8 * fil_num, 8 * fil_num, drop_rate, (3, 1, 1), False)

        self.block_9 = torch.nn.AvgPool3d(kernel_size=(6, 7, 6), stride=3)

        self.dense_ge = nn.Sequential(
            nn.Linear(9, 48),
            nn.ReLU())
        params = torch.ones((1, 9, 337), requires_grad=True).cuda()
        self.params_snp_modifier = nn.Parameter(params)
        self.register_parameter('snp_modifier', self.params_snp_modifier)
        self.dense_snp = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(337, 48),
            nn.ReLU())

        self.gcn_layer_1 = GCN(432)
        self.dense_micro_1 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(432, 256),
            nn.ReLU())
        self.gcn_layer_2 = GCN(256)
        self.dense_micro_2 = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.classify = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(320, 24),
            nn.ReLU(),

            nn.Dropout(drop_rate),
            nn.Linear(24, 2),
        )
        self.classify_micro = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(64, 2),
        )
        self.classify_mri = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256, 2),
        )

    def forward(self, data, label, micro_all, label_all):
        idxes, ge, snp, mri = data[:, :, 0].long(), data[:, :, 1:82], data[:, :, 82:3115], data[:, :, 3115:]
        ge_all, snp_all = micro_all[:, :81], micro_all[:, 81:]
        mri = mri.reshape((data.shape[0], 1, 182, 218, 182))
        ge = ge.reshape((data.shape[0], 9, 9))
        snp = snp.reshape((data.shape[0], 9, 337))
        ge_all = ge_all.reshape(micro_all.shape[0], 9, 9)
        snp_all = snp_all.reshape(micro_all.shape[0], 9, 337)

        mri = self.conv_1(mri)
        mri_tmp = self.block_1_1(mri)
        mri_tmp = self.block_1_2(mri_tmp)
        mri = mri + mri_tmp

        mri_tmp = self.block_3_1(mri)
        mri_tmp = self.block_3_2(mri_tmp)
        mri = self.block_3_3(mri)
        mri = mri + mri_tmp

        mri_tmp = self.block_4_1(mri)
        mri_tmp = self.block_4_2(mri_tmp)
        mri = mri + mri_tmp

        mri_tmp = self.block_5_1(mri)
        mri_tmp = self.block_5_2(mri_tmp)
        mri = self.block_5_3(mri)
        mri = mri + mri_tmp

        mri_tmp = self.block_6_1(mri)
        mri_tmp = self.block_6_2(mri_tmp)
        mri = mri + mri_tmp

        mri_tmp = self.block_7_1(mri)
        mri_tmp = self.block_7_2(mri_tmp)
        mri = self.block_7_3(mri)
        mri = mri + mri_tmp

        mri_tmp = self.block_8_1(mri)
        mri_tmp = self.block_8_2(mri_tmp)
        mri = mri + mri_tmp

        mri = self.block_9(mri)

        ge = self.dense_ge(ge)
        snp = snp * self.params_snp_modifier
        snp = self.dense_snp(snp)
        cons_loss = self.consistency_loss(ge, snp)
        micro = (ge.view(ge.shape[0], -1) + snp.view(snp.shape[0], -1)) / 2
        ge_all = self.dense_ge(ge_all).detach()
        snp_all = (snp_all * self.params_snp_modifier).detach()
        snp_all = self.dense_snp(snp_all).detach()
        micro_all = ((ge_all.view(ge_all.shape[0], -1) + snp_all.view(snp_all.shape[0], -1)) / 2).detach()

        micro, co_loss_1 = self.gcn_layer_1(micro, label, micro_all, label_all)
        micro = self.dense_micro_1(micro)
        micro_all = self.dense_micro_1(micro_all).detach()
        micro, co_loss_2 = self.gcn_layer_2(micro, label, micro_all, label_all)
        micro = self.dense_micro_2(micro)

        mri = mri.view(data.shape[0], -1)
        micro = micro.view(data.shape[0], -1)
        x = torch.cat((micro, mri), dim=1)
        output = self.classify(x)
        output_micro = self.classify_micro(micro)
        output_mri = self.classify_mri(mri)
        return output, output_micro, output_mri, None if co_loss_1 is None else co_loss_1 + co_loss_2, cons_loss

    @staticmethod
    def consistency_loss(feature1, feature2):
        tau = 4
        feature1, feature2 = feature1.view(feature1.shape[0], -1), feature2.view(feature2.shape[0], -1)
        cons_loss = torch.tensor(0.0001).cuda() + \
                    torch.sum(torch.pow((feature1 - feature2) / tau, 2)) / (feature1.shape[0] * 9)
        return cons_loss

    @staticmethod
    def contrastive_loss(feature, label):
        tau = 1
        cont_loss = torch.tensor(0.0001).cuda()
        s = torch.exp(torch.matmul(feature / tau, feature.T / tau))
        mask = torch.repeat_interleave(torch.unsqueeze(label, dim=1), label.shape[0], dim=1)
        mask[:] = 1 - torch.abs(mask[:] - label)
        cont_loss = cont_loss + torch.mean(torch.sum(s * mask, dim=1) / torch.sum(s, dim=1))
        return cont_loss
