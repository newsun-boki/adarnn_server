import os
import pickle
import cv2
import pandas as pd
import numpy as np
import dataset.data_process as data_process
import torch.nn as nn
import torch
import torch.optim as optim
# file_list = {}
# df1=pd.read_pickle("dataset/PRSA_Data_1.pkl")
# data1=df1["Dongsi"]
# df = pd.read_csv("dataset/3_daily_count_1.csv")
# df = np.array(df)
# df = np.reshape(df,(-1,1))
# print(df.shape)
# df = np.reshape(df,-1)
# feat = []
# label_reg = []
# feat = np.array(feat)
# label_reg = np.array(label_reg)
# for i in range(0,df.shape[0] - 25,1):
#     feat = np.append(feat, df[i:i+24])
#     label_reg = np.append(label_reg, df[i + 25])
# feat = np.reshape(feat,(-1,24,1))
# label_reg = np.reshape(label_reg, (-1,))
# print(feat.shape)
# print(label_reg.shape)
# print(len(label_reg))

data_process.load_daily_count_multi_domain('dataset',number_domain = 3,mode = 'tdc')
