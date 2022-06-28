import os
import pickle
import cv2
import pandas as pd

file_list = {}
df1=pd.read_pickle("dataset/PRSA_Data_1.pkl")
data1=df1["Dongsi"]
df = pd.read_csv("dataset/3_daily_count.csv")
print(df)
# # 展示一幅图
# img = file_list['train']['image_data'][0]
# cv2.imwrite("img.png", img)
