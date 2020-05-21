#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:42:21 2018

@author: chris
"""
    



# 把测试集数据整理一下
    
import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import pickle
import random

from bilstm_crf import BiLSTM_CRF
from utility import prepare_sequence, cal_for_acc

def textRec(ans):
    text_re = [''] * (len(ans[1]) + len(ans[2]) + len(ans[3]) + len(ans[4]))
    for i in ans[1]:
        text_re[i] = '\n'
    for i in ans[2]:
        text_re[i] = ' '
    for i in ans[3]:
        text_re[i] = '。'
        
    output = []
    for i in ans[0]:
        for j in i:
            output.append(j)
    for iter, i in enumerate(ans[4]):
        text_re[i] = output[iter]
    return text_re


os.environ["CUDA_VISIBLE_DEVICES"] = '1' # assign GPU


EMBEDDING_DIM = 80
HIDDEN_DIM = 512

with open('../dataset/training_data.plk', 'rb') as f:
    training_data = pickle.load(f)
    train = training_data[0:int(len(training_data) * 0.8)]
    val = training_data[int(len(training_data) * 0.8):]
    
    
# 得到单词到序号的映射
word_to_ix = {'NULL': 0}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for sdir in os.listdir('../dataset/test_data/'):
    path = os.path.join('../dataset/test_data/', sdir)
    with open(path, 'rb') as f:
        stri = pickle.load(f)
        stri = stri[0]
        str_new = []
        # 把stri变成一维的
        for i in stri:
            for j in i:
                str_new.append(j)
        for word in str_new:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            
# 定义标签到序号的映射
START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_ix = {'Disease':0, 'Reason':1, 'Symptom':2, 'Test':3, 'Test_Value':4, 
             'Drug':5, 'Frequency':6, 'Amount':7, 'Method':8, 'Treatment':9,
             'Operation':10, 'Anatomy':11, 'Level':12, 'Duration':13,
             'SideEff':14,'O':15, START_TAG: 16, STOP_TAG: 17}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load('../model_dict/hidim512_1_0.6659692762422823_params.pkl'))
model.cuda() # 调用cuda


base_path = '/home/lingang/chris/knowledge_graph'
# base_path = '/media/chris/D/challenge/knowledge_graph_rename'

file_txt_path = os.path.join(base_path, 'dataset', 'test_data')

result_dict = {}

with torch.no_grad():
    for p in os.listdir(file_txt_path):
        p1= os.path.join(file_txt_path, p)
        output = []
        with open(p1, 'rb') as f:
            ans = pickle.load(f)
            for sentence in ans[0]:
                sentence_in = prepare_sequence(sentence, word_to_ix)
                print(len(sentence_in))
                sentence_in = sentence_in.cuda()
                o = model(sentence_in)
       
                output.append(o[1])
        
        result_dict[str(p[0:-4])] = output
        # 保存 output_list   
        with open('../predict/' + str(p[0:-4]) + '.plk', 'wb') as f:
            pickle.dump(output, f)

# 把这个字典存起来
with open('predict_dict' + '.plk', 'wb') as f:
    pickle.dump(result_dict, f)
    
#%% 上面一块得到n个output文件，用于表示n个文件的预测结果
#   同时用一个字典来保存这些预测结果
    
with open('predict_dict' + '.plk', 'rb') as f:
    result_dict = pickle.load(f)

base_path = '/home/lingang/chris/knowledge_graph'
# base_path = '/media/chris/D/challenge/knowledge_graph_rename'

file_txt_path = os.path.join(base_path, 'dataset', 'test_data')

text_dict = { }

for p in os.listdir(file_txt_path):
    p1 = os.path.join(file_txt_path, p)
    with open(p1, 'rb') as f1:
        text_dict[p[0:-4]] = pickle.load(f1)

# 此时已经有了text_dict 和 result_dict
# 然后根据预测结果，来重构序列，带空格(15)，带换行(15)，带句号100
        
up = { }
for key in text_dict:
    ans = text_dict[key]
    text_re = [''] * (len(ans[1]) + len(ans[2]) + len(ans[3]) + len(ans[4]))
    for i in ans[1]:
        text_re[i] = 100
    for i in ans[2]:
        text_re[i] = 15
    for i in ans[3]:
        text_re[i] = 15   
    output = []
    for j in result_dict[key]:
        for k in j:
            output.append(k)
    for iter, i in enumerate(ans[4]):
        text_re[i] = output[iter]
    up[key] = text_re
    
#%% 根据 up字典得到对应的文本，n个文本
ix_to_tag = ['Disease', 'Reason', 'Symptom', 'Test', 'Test_Value', 'Drug',
             'Frequency', 'Amount', 'Method', 'Treatment', 'Operation',
             'Anatomy', 'Level', 'Duration','SideEff','O'] 
left = -1
mid = []
right = -1
flag = False
record = -1

for key in up:
    file_str = key
    pre = up[key] # pre是一个字符串
    f = open('../upload/' + file_str + '.ann', 'a') 
    for iter, c in enumerate(pre):
        # 找到一个不为15，不为100的索引
        if (c != 15 and c != 100 and flag == False):
            left = iter
            record = c
            flag = True
        if (flag == True):
            if (c == record):
                pass
            elif (c == 100):
                mid.append(iter)
            else:
                right = iter
                if len(mid) == 0:
                    string = str(iter) + '\t' + str(ix_to_tag[record]) + ' ' + str(left) + ' ' + str(right) + '\t entity'
                else:
                    if (right == (mid[-1]+1)):
                        string = str(iter) + '\t' + str(ix_to_tag[record]) + ' ' + str(left) + ' ' + str(mid[-1]) + '\t entity'
                    else:
                        string = str(iter) + '\t' + str(ix_to_tag[record]) + ' ' + str(left) + ' ' + str(mid[0]) + ';' + str(mid[-1] + 1) + ' ' + str(right) + '\t entity'
                print('T' + string)
                f.write('T' + string + '\n')
                mid = []
                left = -1
                right = -1
                record = -1
                flag = False
    f.close()
    











