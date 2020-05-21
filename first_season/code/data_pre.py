#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 15:25:17 2018

@author: chris
"""
#%%
import os

base_path = 'C:\\Users\\sheld\\Desktop\\比赛\\knowledge_graph'
file_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_train2_20181022', '0.txt')
file_ano_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_train2_20181022', '0.ann')
print(base_path)
print(file_path)
print(file_ano_path)


#%%
text_str = ''
with open(file_path, 'r', encoding='UTF-8') as f:
    text_str = f.read()

with open(file_ano_path, 'r', encoding='UTF-8') as f:
    ano_str = f.read()
print(text_str)
print(ano_str)


#%%
anotation = ano_str.split('\n')
tag_to_ix = {'Disease':0, 'Reason':1, 'Symptom':2, 'Test':3, 'Test_Value':4, 
             'Drug':5, 'Frequency':6, 'Amount':7, 'Method':8, 'Treatment':9,
             'Operation':10, 'Anatomy':11, 'Level':12, 'Duration':13,
             'SideEff':14,'O':15}
print(tag_to_ix)

#%%
# 获取一个文本的annotation，一个列表
annotation_list = ['O'] * len(text_str) 
count = 0
for label in anotation:
    if label != '':
        x = label.split('\t')
        x = x[1]
        if (';' in x):
            xx = x.split(';')
            # print(xx)
            xx_l = xx[0].split(' ')
            xx_r = xx[1].split(' ')
            for i in range(int(xx_l[1]), int(xx_l[2])):
                annotation_list[i] = xx_l[0]
            for i in range(int(xx_r[0]), int(xx_r[1])):
                annotation_list[i] = xx_l[0]
            count += 1
        else:
            xx = x.split(' ')
            # print(xx)
            for i in range(int(xx[1]), int(xx[2])):
                annotation_list[i] = xx[0]
            count += 1
print(count)


#%%
# 对于文本字符列表和标签字符列表，去除换行，空格，并用句号来分割句子
text_s = [i for i in text_str]
annotation_list

# 把换行去了
while ('\n' in text_s):
    i = text_s.index('\n')
    del(text_s[i])
    del(annotation_list[i])
    
while (' ' in text_s):
    i = text_s.index(' ')
    del(text_s[i])
    del(annotation_list[i])

#%根据句号来分割句子
tt = text_s.copy()
tt_ = annotation_list.copy()
train_data = []
train_label = []
while ('。' in tt):
    i = tt.index('。') #找到当前第一个句号
    train_data.append(tt[0 : i])
    train_label.append(tt_[0 : i])
    tt = tt[i + 1:]
    tt_ = tt_[i+1:]
#%%
print(len(train_data))

###############################################################################
#   
# 
# 上面的全是测试代码
#    
#    
###############################################################################
#%%
# 一个函数: 根据两个文件 -> 训练样本的形式, train_data

def file2data(path_txt, path_ano):
    text_str = ''
    ano_str = ''
    
    with open(path_txt, 'r', encoding='UTF-8') as f:
        text_str = f.read()
    with open(path_ano, 'r', encoding='UTF-8') as f:
        ano_str = f.read()
        
    annotation_list = ['O'] * len(text_str) 

    # 把ano文件的数据解析出来
    for label in ano_str.split('\n'):
        if label != '':
            x = label.split('\t')
            x = x[1]
            if (';' in x):
                xx = x.split(';')
                # print(xx)
                xx_l = xx[0].split(' ')
                xx_r = xx[1].split(' ')
                for i in range(int(xx_l[1]), int(xx_l[2])):
                    annotation_list[i] = xx_l[0]
                for i in range(int(xx_r[0]), int(xx_r[1])):
                    annotation_list[i] = xx_l[0]
            else:
                xx = x.split(' ')
                # print(xx)
                for i in range(int(xx[1]), int(xx[2])):
                    annotation_list[i] = xx[0]
    text_list = [i for i in text_str]
    
#    # 把换行,空格去了
#    while ('\n' in text_list):
#        i = text_list.index('\n')
#        del(text_list[i])
#        del(annotation_list[i])
#        
#    while (' ' in text_list):
#        i = text_list.index(' ')
#        del(text_list[i])
#        del(annotation_list[i])
    # 根据句号来分割句子
    tt = text_list.copy()
    tt_ = annotation_list.copy()
    train_data = []
    while ('。' in tt):
        i = tt.index('。') #找到当前第一个句号
        if (i > 0):
            train_data.append((tt[0 : i], tt_[0 : i]))
        tt = tt[i + 1:]
        tt_ = tt_[i+1:]
    return train_data

#%%
# 把文本转化为 训练所需要的格式
import os

base_path = 'C:\\Users\\sheld\\Desktop\\比赛\\knowledge_graph'
file_txt_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_train2_20181022', 'txt')
file_ano_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_train2_20181022', 'ann')
print(file_ano_path)
#%%
training = []
for p in os.listdir(file_txt_path):
    p1= os.path.join(file_txt_path, p)
    p2 = os.path.join(file_ano_path, p[: -4] + '.ann')
    training.extend(file2data(p1, p2))


# 使用pickle 保存训练数据
import pickle
with open('..\\dataset\\training_data_with_ns.plk', 'wb') as f:
    pickle.dump(training, f)

#%%
print(len(training))
print(training[0])


#%%
with open('..\\dataset\\training_data.plk', 'rb') as f:
    ans = pickle.load(f)
#%%

#  得到测试集数据，用于预测
import os
text_str = ''
ano_str = ''
base_path = 'C:\\Users\\sheld\\Desktop\\比赛\\knowledge_graph'
file_txt_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_test_a_20181022')
path_txt = os.path.join(file_txt_path, '7.txt')

with open(path_txt, 'r') as f:
    text_str = f.read()

text_list = [i for i in text_str]

# # 得到所有换行和空格的索引， '\n'
# index_n = []
# index_space = []
# index_char = []
# for iter, char in enumerate(text_list):
#     if (char == '\n'):
#         index_n.append(iter)
#     elif (char == ' '):
#         index_space.append(iter)
#     else:
#         index_char.append(iter)

# # 删除所有空格和换行
# while ('\n' in text_list):
#     i = text_list.index('\n')
#     del(text_list[i])
    
# while (' ' in text_list):
#     i = text_list.index(' ')
#     del(text_list[i])

#

# text_re = [''] * len(text_str)
# for i in index_n:
#     text_re[i] = '\n'
# for i in index_space:
#     text_re[i] = ' '
# for iter, i in enumerate(index_char):
#     text_re[i] = text_list[iter]

#%%
# 测试文件转变为预测用的数据
def file2dataFortest(path_txt):
    # 得到txt序列
    with open(path_txt, 'r', encoding='UTF-8') as f:
        text_str = f.read()
    text_list = [i for i in text_str]
    
    # 得到字符和句号的索引
    index_char = []
    index_o = []
    for iter, char in enumerate(text_list):
        if (char == '。'):
            index_o.append(iter)
        else:
            index_char.append(iter)
    
    # # 删除所有空格和换行
    # while ('\n' in text_list):
    #     i = text_list.index('\n')
    #     del(text_list[i])
        
    # while (' ' in text_list):
    #     i = text_list.index(' ')
    #     del(text_list[i])
    
    # 用句号来切分句子
    tt = text_list.copy()
    test_data = []
    while ('。' in tt):
        i = tt.index('。') #找到当前第一个句号
        if (i > 0):
            test_data.append(tt[0 : i])
        tt = tt[i + 1:]
    if (len(tt) > 0):
        test_data.append(tt)

        
    return (test_data, index_o, index_char)

#二维列表展开

    
#%%
# 对序列进行重构
def textRec(ans):
    text_re = [''] * (len(ans[1]) + len(ans[2]))
    for i in ans[1]:
        text_re[i] = '。'
        
    output = []
    for i in ans[0]:
        for j in i:
            output.append(j)
    for iter, i in enumerate(ans[2]):
        text_re[i] = output[iter]
    return text_re

#%%
#把文本转化为 训练所需要的格式
import os
import pickle

base_path = 'C:\\Users\\sheld\\Desktop\\比赛\\knowledge_graph'
file_txt_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_test_a_20181022')


for p in os.listdir(file_txt_path):
     
    p1= os.path.join(file_txt_path, p)
    testing = file2dataFortest(p1)
    
    with open('../dataset/test_data/' + p[0:-4] + '.plk', 'wb') as f:
        pickle.dump(testing, f)



#%%
# 看一下测试数据
with open('..\\dataset\\test_data\\7.plk',  'rb') as f:
    ans = pickle.load(f)

print(len(ans))
print(len(ans[0]), len(ans[1]),len(ans[2]))


#%%
print(ans[2])

#%%
ans_path = 'C:\\Users\\sheld\\Desktop\\比赛\\knowledge_graph\\rawdata\\ruijin_round1_test_a_20181022\\7.txt'
with open(ans_path, 'r', encoding='UTF-8') as f:
    ans = f.read()
print(len(ans))


#%%
7820 + 138

#%%
