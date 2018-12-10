import os
import pickle
import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, Concatenate, Conv1D, Dense, Embedding,
                          Flatten, Input, MaxPooling1D, Dropout, BatchNormalization)
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras import backend as K

base_path = os.path.dirname(os.getcwd())
with open(os.path.join(base_path, 'dataset', 'train', 'X_train_150_0-120.plk'), 'rb') as f:
    X_train = pickle.load(f)
    X_train_word = X_train[:, :, 0]
    X_train_pos1 = X_train[:, :, 1]
    X_train_pos2 = X_train[:, :, 2]
    print('训练集长度为：', X_train.shape)

with open(os.path.join(base_path, 'dataset', 'train', 'Y_train_150_0-120.plk'), 'rb') as f:
    Y_train = pickle.load(f)

with open(os.path.join(base_path, 'dataset', 'val', 'X_val_150_0-120.plk'), 'rb') as f:
    X_val = pickle.load(f)
    X_val_word = X_val[:, :, 0]
    X_val_pos1 = X_val[:, :, 1]
    X_val_pos2 = X_val[:, :, 2]
    print('验证集长度为：', X_val.shape)

with open(os.path.join(base_path, 'dataset', 'val', 'Y_val_150_0-120.plk'), 'rb') as f:
    Y_val = pickle.load(f)

with open(os.path.join(base_path, 'dataset', 'word_to_ix.plk'), 'rb') as f:
    word_to_ix = pickle.load(f)

def pre_t(y_true, y_pred):
    y_t = y_true[:, 1]
    y_p = y_pred[:, 1]
    return K.sum(K.round(K.clip(y_p, 0, 1))) # 预测为正的个数

def real_t(y_true, y_pred):
    y_t = y_true[:, 1]
    y_p = y_pred[:, 1]
    return K.sum(K.round(K.clip(y_t, 0, 1))) # 实际为正的个数

def inter(y_true, y_pred):
    y_t = y_true[:, 1]
    y_p = y_pred[:, 1]
    return K.sum(K.round(K.clip(y_t * y_p, 0, 1))) # 为1的交集个数

def F1(y_true, y_pred):
    K.set_epsilon(1e-12)
    predicted_positives = pre_t(y_true, y_pred) # 预测为正的个数
    possible_positives = real_t(y_true, y_pred) # 实际为正的个数
    intersect = inter(y_true, y_pred)  # 为1的交集个数
    P = intersect / (predicted_positives + K.epsilon())
    R = intersect / (possible_positives + K.epsilon())
    return 2*P*R/(P+R+K.epsilon())


sentence_length = 150
voca_size = len(word_to_ix)
word_embed_dim = 64
pos_size = 305
pos_embed_dim = 5


# 句子长度为batch * 150的矩阵
inputs_word = Input(shape=(sentence_length,))
inputs_pos1 = Input(shape=(sentence_length,))
inputs_pos2 = Input(shape=(sentence_length,))

x1 = Embedding(voca_size, word_embed_dim, input_length=sentence_length)(inputs_word)
x2 = Embedding(pos_size, pos_embed_dim, input_length=sentence_length)(inputs_pos1)
x3 = Embedding(pos_size, pos_embed_dim, input_length=sentence_length)(inputs_pos2)

x = Concatenate(axis=-1)([x1, x2, x3])

# 以张量为参数返回一个张量
x = Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', kernel_initializer='he_normal')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size = 2, strides=2, padding='valid')(x)

x = Conv1D(filters=128, kernel_size=3, strides=1, padding='valid', kernel_initializer='he_normal')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size = 2, strides=2, padding='valid')(x)

x = Conv1D(filters=256, kernel_size=3, strides=1, padding='valid', kernel_initializer='he_normal')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size = 2, strides=2, padding='valid')(x)

x = Conv1D(filters=256, kernel_size=3, strides=1, padding='valid', kernel_initializer='he_normal')(x)
#x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size = 2, strides=2, padding='valid')(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x) # 有关系或者没关系

# 创建模型, 每一轮保存模型
model = Model(inputs=[inputs_word, inputs_pos1, inputs_pos2], outputs=outputs)
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy', F1, pre_t, real_t, inter])

filepath = os.path.join(base_path, 'checkpoint', 'M_150_100.{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{val_acc:.4f}.h5')
checkpoint = ModelCheckpoint(filepath)

# 计算正负样本比例
porpotion = 1
tt = 0
ff = 0
for i in range(len(Y_train)):
    if (Y_train[i][1] == 1):
        tt += 1
    else:
        ff += 1
porpotion = ff/tt
print('正样本个数为：%d' %tt)
print('负样本个数为：%d' %ff)
print('负样本/正样本比例为：%0.4f' %(porpotion))

# 对训练样本设置权重
print('正在设置样本权重...')
sample_weight = np.ones([len(Y_train),])
for i in range(len(Y_train)):
    # 实体对有关
    if Y_train[i][1] == 1:
        sample_weight[i] = porpotion

class F1_val(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        real_t = logs.get('val_real_t')
        pre_t = logs.get('val_pre_t')
        inter = logs.get('val_inter')
        P = inter / (real_t + 1e-12)
        R = inter / (pre_t + 1e-12)
        f1_score = 2*P*R/(P+R+1e-12)
        print('* 验证集F1为:%0.6f\tinter_avg:%0.6f\tpre_t_avg:%0.6f\treal_t_avg:%0.6f' %(f1_score, inter, pre_t, real_t))
        
f1_val = F1_val()
# 开始训练
history = model.fit([X_train_word, X_train_pos1 + 150, X_train_pos2 + 150], 
            Y_train, 
            batch_size=32,
            epochs=32, 
            verbose=1,
            validation_data = ([X_val_word, X_val_pos1 + 150, X_val_pos2 + 150], Y_val),
            # sample_weight = sample_weight,
            callbacks=[checkpoint, f1_val])  # 开始训练

