import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import datetime
from base.loss_transfer import TransferLoss

class data_loader(Dataset):
    def __init__(self, df_feature, df_label, df_label_reg, t=None):

        # assert len(df_feature) == len(df_label)
        assert len(df_feature) == len(df_label_reg)

        # df_feature = df_feature.reshape(df_feature.shape[0], df_feature.shape[1] // 6, df_feature.shape[2] * 6)
        self.df_feature=df_feature
        self.df_label=df_label
        self.df_label_reg = df_label_reg

        self.T=t
        self.df_feature=torch.tensor(
            self.df_feature, dtype=torch.float32)
        self.df_label=torch.tensor(
            self.df_label, dtype=torch.float32)
        self.df_label_reg=torch.tensor(
            self.df_label_reg, dtype=torch.float32)

    def __getitem__(self, index):
        sample, target, label_reg =self.df_feature[index], self.df_label[index], self.df_label_reg[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target, label_reg

    def __len__(self):
        return len(self.df_feature)

def process_data(df):
    df = np.reshape(df,-1)
    feat = []
    label_reg = []
    feat = np.array(feat)
    label_reg = np.array(label_reg)
    for i in range(0,df.shape[0] - 25,1):#通过前24天预测1天
        feat = np.append(feat, df[i:i+24])
        label_reg = np.append(label_reg, df[i + 25])
    feat = np.reshape(feat,(-1,24,1))
    label_reg = np.reshape(label_reg, (-1,))
    return feat,label_reg

def get_dataset_statistic(df, start_index, end_index):
    feat,label_reg = process_data(df)
    referece_start_index = 0
    referece_end_index = len(label_reg)
    assert (start_index - referece_start_index >= 0)
    assert (end_index - referece_end_index) <= 0
    assert (end_index - start_index) >= 0
    index_start = start_index - referece_start_index
    index_end = end_index - referece_end_index
    feat=feat[index_start: index_end + 1]
    label_reg=label_reg[index_start: index_end + 1]
    feat=feat.reshape(-1, feat.shape[2])
    mu_train=np.mean(feat, axis=0)
    sigma_train=np.std(feat, axis=0)

    return mu_train, sigma_train


def create_dataset(df, start_index, end_index, mean=None, std=None):
    feat,label_reg = process_data(df)
    referece_start_index = 0
    referece_end_index = len(label_reg)
    assert (start_index - referece_start_index >= 0)
    assert (end_index - referece_end_index) <= 0
    assert (end_index - start_index) >= 0
    feat=feat[start_index: end_index + 1]
    label_reg=label_reg[start_index: end_index + 1]
    print(end_index + 1)
    label = np.arange(label_reg.shape[0])# not use
    # ori_shape_1, ori_shape_2=feat.shape[1], feat.shape[2]
    # feat=feat.reshape(-1, feat.shape[2])
    # feat=(feat - mean) / std
    # feat=feat.reshape(-1, ori_shape_1, ori_shape_2)
    print('feat.shape',feat.shape)
    print('label_reg.shape',label_reg.shape)
    return data_loader(feat, label, label_reg)

def get_count_data_statistic(data_file, start_index, end_index):
    df = pd.read_csv(data_file)
    df = np.array(df)
    mean_train, std_train =get_dataset_statistic(
        df, start_index, end_index)
    return mean_train, std_train

def get_count_data(data_file, start_time, end_time, batch_size, shuffle=True, mean=None, std=None):
    df = pd.read_csv(data_file)
    df = np.array(df)

    dataset=create_dataset(df, start_time,
                             end_time, mean=mean, std=std)
    train_loader=DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader


def get_split_time(num_domain=2, mode='pre_process', data_file = None, dis_type = 'coral'):
    spilt_index = {
        '2': [(0, 200), (200, 300)],
        '5':[(0,46),(47,211),(212,241),(242,270),(271,300)]
    }
    if mode == 'pre_process':
        return spilt_index[str(num_domain)]
    if mode == 'tdc':
        return TDC(num_domain, data_file,  dis_type = dis_type)
    else:
        print("error in mode")


def TDC(num_domain, data_file, dis_type = 'coral'):
    num_day = 300
    split_N = 10 #10 before
    df = pd.read_csv(data_file)
    df = np.array(df)
    df = np.reshape(df,(-1,1))
    feat = df
    feat=torch.tensor(feat, dtype=torch.float32)
    feat = feat.cuda()
    # num_day_new = feat.shape[0]

    selected = [0, 10]
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    start = 0
    
    if num_domain in [2, 3, 5, 7, 10]:
        while len(selected) -2 < num_domain -1:
            distance_list = []
            for can in candidate:
                selected.append(can)
                selected.sort()
                dis_temp = 0
                for i in range(1, len(selected)-1):
                    for j in range(i, len(selected)-1):
                        index_part1_start = start + math.floor(selected[i-1] / split_N * num_day) 
                        index_part1_end = start + math.floor(selected[i] / split_N * num_day)
                        feat_part1 = feat[index_part1_start: index_part1_end]
                        index_part2_start = start + math.floor(selected[j] / split_N * num_day)
                        index_part2_end = start + math.floor(selected[j+1] / split_N * num_day)
                        feat_part2 = feat[index_part2_start:index_part2_end]
                        criterion_transder = TransferLoss(loss_type= dis_type, input_dim=feat_part1.shape[1])
                        dis_temp += criterion_transder.compute(feat_part1, feat_part2)
                distance_list.append(dis_temp)
                selected.remove(can)
            can_index = distance_list.index(max(distance_list))
            selected.append(candidate[can_index])
            candidate.remove(candidate[can_index]) 
        selected.sort()
        res = []  
        for i in range(1,len(selected)):
            if i == 1:
                sel_start_time = int(num_day / split_N * selected[i - 1])
            else:
                sel_start_time = int(num_day / split_N * selected[i - 1])+1
            sel_end_time = int(num_day / split_N * selected[i])
            res.append((sel_start_time, sel_end_time))
        return res
    else:
        print("error in number of domain")