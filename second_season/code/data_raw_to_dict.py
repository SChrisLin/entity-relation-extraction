'''
    本脚本将原始数据集转化为一个字典，便于后续的样本处理。
'''
import numpy as np 
import matplotlib.pyplot as plt
import os
import re
import pickle

def annstr_to_list(ann_str):
    '''
        输入：从某个ann文件中读进来的字符串.
        输出：两个list，一个记录了ann文件中所有的entity;另一个记录了ann文件中所有的relation.
    '''
    # 用\n切分字符串
    en_r_list = ann_str.split('\n')
    # 将list切分为实体list和关系list
    entity_list = []
    re_list = []
    for i in en_r_list:
        if (i != ''):
            if (i[0] == 'T'):
                entity_list.append(i)
            elif (i[0] == 'R'):
                re_list.append(i)
            else:
                print('error')
    return entity_list, re_list

def entity_list_to_dict(entity_list):
    '''
        输入：一个list，记录了ann文件中的entity
        输出：一个dict.
            字典格式为：
                {'entity_name':{'type': 'Disease', 'pos':[1,2,3]} }
    '''
    en_dict = {}
    for i in entity_list:
        list_s = i.split('\t')
        pos_s = re.split('[ ,;]', list_s[1])
        entity_name = list_s[0]
        entity_type = pos_s[0]
        if (len(pos_s) == 3):
            left = int(pos_s[1])
            right = int(pos_s[2])
            entity_pos = list(range(left, right))
        else:
            left = int(pos_s[1])
            left_ = int(pos_s[2])
            right = int(pos_s[3])
            right_ = int(pos_s[4])
            entity_pos = list(range(left, left_)) + list(range(right, right_))
        entity_dict = {}
        entity_dict['type'] = entity_type
        entity_dict['pos'] = entity_pos
        en_dict[entity_name] = entity_dict
    return en_dict

def relation_list_to_dict(re_list):
    '''
        输入：一个list，记录了一个ann文件中的realtion
        输出：一个dict
            字典格式为：
                {'relation_name':{'type': 'Drug_disease', 'entity':[T1, T2]} }
    '''
    re_dict = {}
    for i in re_list:
        list_s = i.split('\t')
        pos_s = re.split('[ ,:]', list_s[1])
        relation_name = list_s[0]
        relation_type = pos_s[0]
        relation_dict = {}
        relation_dict['type'] = relation_type
        relation_dict['entity'] = [pos_s[2], pos_s[4]]
        re_dict[relation_name] = relation_dict
    return re_dict

def raw_train_to_dict(path_train):
    '''
        根据训练集路径，将train数据集解析成字典，字典格式为：
            {'file_name':
                {'text'    : 'abc...', 
                 'entity'  : {'entity_name':{'type': 'Disease', 'pos':[1,2,3]} },
                 'relation': {'relation_name':{'type': 'Drug_disease', 'entity':[T1, T2]}}}}
        例如，
            {'0':
                {'text'    : 'abc', 
                 'entity'  : {'T1':{'type': 'Disease', 'pos':[1,2,3]}, 
                              'T2':{'type': 'Disease', 'pos':[4,5,6]} },
                 'relation': {'R1':{'type': 'Drug_disease', 'entity':[T1, T2]}},
                              'R2':{'type': 'Drug_disease', 'entity':[T3, T4]}}}}
    '''
    train_data_dict = {}
    for name in os.listdir(os.path.join(path_train, 'txt')):
        txt_dir = os.path.join(path_train, 'txt', name)
        ann_dir = os.path.join(path_train, 'ann', name[:-4] + '.ann')
        with open(txt_dir, 'r', encoding='utf-8') as f:
            txt_str = f.read()
        with open(ann_dir, 'r', encoding='utf-8') as f:
            ann_str = f.read()
        # 根据ann_str得到所有的实体以及关系
        entity_list, re_list = annstr_to_list(ann_str)
        entity_dict = entity_list_to_dict(entity_list)
        relation_dict = relation_list_to_dict(re_list)
        dd = {}
        dd['text'] = txt_str
        dd['entity'] = entity_dict
        dd['relation'] = relation_dict
        train_data_dict[name[:-4]] = dd
    return train_data_dict

def raw_test_to_dict(path_test):
    '''
        将test数据集解析成字典，字典格式为：
            {'file_name':
                {'text'    : 'abc...', 
                 'entity'  : {'entity_name':{'type': 'Disease', 'pos':[1,2,3]} }}}
        例如，
            {'0':
                {'text'    : 'abc', 
                 'entity'  : {'T1':{'type': 'Disease', 'pos':[1,2,3]} }}
    '''
    test_data_dict = {}
    for name in os.listdir(os.path.join(path_test, 'txt')):
        txt_dir = os.path.join(path_test, 'txt', name)
        ann_dir = os.path.join(path_test, 'ann', name[:-4] + '.ann')
        with open(txt_dir, 'r', encoding='utf-8') as f:
            txt_str = f.read()
        with open(ann_dir, 'r', encoding='utf-8') as f:
            ann_str = f.read()
        # 根据ann_str得到所有的实体以及关系
        entity_list = []
        en_r_list = ann_str.split('\n')
        for i in en_r_list:
            if (i != ''):
                entity_list.append(i)
        entity_dict = entity_list_to_dict(entity_list)
        dd = {}
        dd['text'] = txt_str
        dd['entity'] = entity_dict
        test_data_dict[name[:-4]] = dd
    return test_data_dict

if __name__ == "__main__":

    # 一些基本路径，请注意当前路径一定要为code文件夹
    base_path = os.path.dirname(os.getcwd())
    path_rawdata = os.path.join(base_path, 'rawdata')
    path_train = os.path.join(path_rawdata, 'train')
    path_test = os.path.join(path_rawdata, 'test')

    '''得到训练数据的字典, 并保存到文件
        # 字典格式如下：
        {'0':
            {'text'    : 'abc', 
                'entity'  : {'T1':{'type': 'Disease', 'pos':[1,2,3]}, 
                            'T2':{'type': 'Disease', 'pos':[4,5,6]} },
                'relation': {'R1':{'type': 'Drug_disease', 'entity':[T1, T2]}},
                            'R2':{'type': 'Drug_disease', 'entity':[T3, T4]}}}}
    '''
    print('正在将训练集样本转化为字典：train_data_dict.plk...')
    train_dict = raw_train_to_dict(path_train)
    save_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset', 'train_data_dict.plk')
    # 保存文件
    with open(save_path, 'wb') as f:
        pickle.dump(train_dict, f)
    print('训练集样本处理完成')
    '''得到测试数据的字典, 并保存到文件
        # 字典格式如下：
        {'0':
            {'text'    : 'abc', 
                'entity'  : {'T1':{'type': 'Disease', 'pos':[1,2,3]}, 
                            'T2':{'type': 'Disease', 'pos':[4,5,6]}}}}
    '''
    print('正在将测试集样本转化为字典：test_data_dict.plk...')
    text_dict = raw_test_to_dict(path_test)
    save_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset', 'test_data_dict.plk')
    # 保存文件
    with open(save_path, 'wb') as f:
        pickle.dump(text_dict, f)
    print('测试集样本处理完成')
