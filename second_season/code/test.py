'''
    这个脚本主要用来测试一些输出是否正确
'''
import os
import pickle

base_path = os.path.dirname(os.getcwd())
path_x = os.path.join(base_path, 'dataset', 'train', 'X_train_150_0-120.plk')
path_y = os.path.join(base_path, 'dataset', 'train', 'Y_train_150_0-120.plk')

with open(path_x, 'rb') as f:
    X = pickle.load(f)
with open(path_y, 'rb') as f:
    Y = pickle.load(f)

print(len(X))
print(len(Y))

print(X[1], '\n', Y[1])


path_word_to_ix = os.path.join(base_path, 'dataset', 'word_to_ix.plk')
with open(path_word_to_ix, 'rb') as f:
    word_to_ix = pickle.load(f)

len(word_to_ix)