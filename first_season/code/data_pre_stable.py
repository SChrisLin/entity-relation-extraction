
import pickle
import os

def file2data(path_txt, path_ano):
    ''' 
        对于训练数据，把txt和ano文件夹中的文本转化为训练样本所需格式。
        输出为一个列表.列表中包含n个元组，表示n个数据。
        每个元组中两个字符串列表，一个是文本，另一个是对应的标签。
    '''
    text_str = ''
    ano_str = ''
    with open(path_txt, 'r', encoding='UTF-8') as f:
        text_str = f.read()
    with open(path_ano, 'r', encoding='UTF-8') as f:
        ano_str = f.read()
    annotation_list = ['O'] * len(text_str) 
    # 把ano文件的数据解析出来, 得到annotation_list
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

def file2dataFortest(path_txt):
    '''
        将原始测试转变为预测用的数据。
        输出一个元组，包含3个元素，第一个元素为n个句子，第二个元素为所有句号的索引，
        第三个元素为所有字符的索引。
    '''
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

def textRec(ans):
    '''
        对序列进行重构:将测试数据转化为一串包含句号的字符串。
    '''
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

def data_process():
    # 把训练数据转化为需要的格式
    base_path = os.path.dirname(os.getcwd())
    file_txt_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_train2_20181022', 'txt')
    file_ano_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_train2_20181022', 'ann')
    # training 数据是一个列表
    training = []
    for p in os.listdir(file_txt_path):
        p1= os.path.join(file_txt_path, p)
        p2 = os.path.join(file_ano_path, p[: -4] + '.ann')
        training.extend(file2data(p1, p2))
    # 使用pickle 保存训练数据
    with open(os.path.join(base_path, 'dataset', 'training_data_with_ns.plk'), 'wb') as f:
        pickle.dump(training, f)
    # 把测试数据转化为需要的格式
    file_txt_path = os.path.join(base_path, 'rawdata', 'ruijin_round1_test_a_20181022')
    for p in os.listdir(file_txt_path):
        p1= os.path.join(file_txt_path, p)
        testing = file2dataFortest(p1)
        with open(os.path.join(base_path, 'dataset', 'test_data', p[0:-4] + '.plk'), 'wb') as f:
            pickle.dump(testing, f)

if __name__ == '__main__':
    data_process()
    
