import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def sentence_to_num(sentence, to_ix):
    '''
        根据to_ix字典，将字符串转化为，数字列表.
    '''
    idxs = []
    for char in sentence:
        if char in list(to_ix.keys()):
            idxs.append(to_ix[char])
        else:
            idxs.append(0)
    return idxs

# 一些路径
base_path = os.path.dirname(os.getcwd())
path_train_dict = os.path.join(base_path, 'dataset', 'train_data_dict.plk')
path_train_list = os.path.join(base_path, 'dataset', 'train_list_150_0-120.plk')
path_word_to_ix = os.path.join(base_path, 'dataset', 'word_to_ix.plk')

# 导入训练集字典
with open(path_train_dict, 'rb') as f:
    train_data_dict = pickle.load(f)

# 导入word_to_ix
word_to_ix = {'NULL': 0}
# 判断一个文件存不存在
if not os.path.exists(path_word_to_ix):
    # 构造word_to_index
    for key in train_data_dict.keys():
        for char in train_data_dict[key]['text']:
            if char not in word_to_ix:
                word_to_ix[char] = len(word_to_ix)
    with open(path_word_to_ix, 'wb') as f:
        pickle.dump(word_to_ix, f)
else:
    with open(path_word_to_ix, 'rb') as f:
        word_to_ix = pickle.load(f)

# 导入train_list
with open(path_train_list, 'rb') as f:
    train_list = pickle.load(f)

# 将train_list中的字母转化为数字
for iter, list_ in enumerate(train_list):
    list_[0][0] = sentence_to_num(list_[0][0], word_to_ix)
    print('\r正在将字符转化为数字： %d/%d,  %0.6f' %(iter+1, len(train_list), (iter+1)/len(train_list)), end='', flush=True)

print('\n 转换完毕！\n')

X_train = np.zeros([len(train_list), 150, 3], dtype=np.int16)
Y_train = np.zeros([len(train_list), 2], dtype=np.int8)
# 将list变成int16类型的array
for iter, ls in enumerate(train_list):
    X_train[iter] = np.array(ls[0], dtype=np.int16).T
    Y_train[iter] = np.array(ls[1], dtype=np.int8)
    print('\r正在构造训练样本： %d/%d, %0.4f' %(iter+1, len(train_list), (iter+1)/len(train_list)), end='', flush=True)

print('\n构造完毕！')
print('正在保存训练样本...')
# 把X，y存起来
with open(os.path.join(base_path, 'dataset', 'train', 'X_train_150_0-120.plk'), 'wb') as f:
    pickle.dump(X_train, f)
with open(os.path.join(base_path, 'dataset', 'train', 'Y_train_150_0-120.plk'), 'wb') as f:
    pickle.dump(Y_train, f)
print('保存完毕！')
