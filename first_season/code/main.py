'''
    待更新...
'''

import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import pickle
import random

from bilstm_crf import BiLSTM_CRF
from utility import prepare_sequence, cal_for_acc



os.environ["CUDA_VISIBLE_DEVICES"] = '1' # assign GPU


EMBEDDING_DIM = 80
HIDDEN_DIM = 512

with open('../dataset/training_data.plk', 'rb') as f:
    training_data = pickle.load(f)
    # random.shuffle(training_data)
    train = training_data[0:int(len(training_data) * 0.8)]
    val = training_data[int(len(training_data) * 0.8):]
#%%    
    
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

#%%        
# 定义标签到序号的映射
START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_ix = {'Disease':0, 'Reason':1, 'Symptom':2, 'Test':3, 'Test_Value':4, 
             'Drug':5, 'Frequency':6, 'Amount':7, 'Method':8, 'Treatment':9,
             'Operation':10, 'Anatomy':11, 'Level':12, 'Duration':13,
             'SideEff':14,'O':15, START_TAG: 16, STOP_TAG: 17}

#%%
def f1_in_batch_data(valdata, model):
    inter = 0
    out_entity_num = 0
    target_entity_num = 0
    F1 = 0
    loss_ = 0
    average_loss = 0
    count = 0
    print('\n')
    for sentence, tags in valdata:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        sentence_in = sentence_in.cuda()
        targets = torch.cuda.LongTensor([tag_to_ix[t] for t in tags])
        targets = targets.cuda()
        output = model(sentence_in)
        I, O, T = cal_for_acc(output[1], targets)
        inter += I
        out_entity_num += O
        target_entity_num += T
        P = inter / (out_entity_num + 1e-18)
        R = inter / (target_entity_num + 1e-18)
        F1 = 2 * P * R / (P + R + 1e-18)
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss_ += (loss.item() / len(sentence))
        count += 1
        average_loss = loss_ / (count + 1e-18)
        print('\rtesting:%d/%d\t loss:%0.5f \t' %(count, len(valdata), average_loss), end='', flush=True)
    return F1, average_loss

#%%
        
# 定义模型和优化器
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
# model.load_state_dict(torch.load('../model_dict/params.pkl'))
model.cuda() # 调用cuda
#%%
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(),lr = 1e-4, amsgrad=True, weight_decay=1e-5)

#%%
# 开始训练
for epoch in range(0, 10):  # again, normally you would NOT do 300 epochs, it is toy data
    inter = 0
    out_entity_num = 0
    target_entity_num = 0
    loss_ = 0
    for iter, (sentence, tags) in enumerate(train):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        sentence_in = sentence_in.cuda()
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        targets = targets.cuda()
        
        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss_ += (loss.item()/len(sentence))
        
        with torch.no_grad():
            output = model(sentence_in)
            I, O, T = cal_for_acc(output[1], targets)
            inter += I
            out_entity_num += O
            target_entity_num += T
            P = inter / (out_entity_num + 1e-18)
            R = inter / (target_entity_num + 1e-18)
            F1 = 2 * P * R / (P + R + 1e-18)
        
        print('\rtraining:%d/%d\t loss:%0.5f \t P:%0.5f \t R:%0.5F \t F1:%0.5f' %(iter, len(train), loss_/(iter + 1), P, R, F1), end='',flush=True)
        
        #print('training:%d/%d\t loss:%0.5f ' %(iter, len(train), loss_/(iter+1)))


        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
        
    # val data accuracy
    with torch.no_grad():
        F1, avg_loss = f1_in_batch_data(val, model)
        print('\n val_acc: \t epoch: %d \tF1 score: %0.5f \n' %(epoch, F1))
        torch.save(model.state_dict(), '../model_dict/hidim512_fulldict_' + str(epoch) + '_'  + str(F1) + '_params.pkl')
        