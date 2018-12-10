'''
    本文档将train_data_dict.plk文件解析出来.
    list中一个元素，像这样：
        [
            [
                ['对','中','国','成','人','2','型','糖','尿','病','H','B','A','1','C',' ',' ','。','目','标'],
                [-5,  -4,  -3,  -2,  -1,  0,   1,  2,   3,   4,   5,  6,  7, 8,  9,  10, 11,  12,  13, 14],
                [-14, -13, -12, -11,-10, -9, -8,  -7,  -6,  -5,  -4, -3, -2,-1,  0,  1,   2,   3,  4,   5]
            ],
            [0, 1]
        ]
'''
import os
import pickle
import random
import numpy as np

# 基本路径
base_path = os.path.dirname(os.getcwd())
# 数据集路径
data_dict_path = os.path.join(base_path, 'dataset')
# 训练集字典路径
train_dict_path = os.path.join(data_dict_path, 'train_data_dict.plk')


# 导入训练集字典,验证集字典和测试集字典
with open(train_dict_path, 'rb') as f:
    train_dict = pickle.load(f)

def get_realtion(type1, type2):
    '''
        得到两个实体关系的种类
    '''
    type_all = ['Test_Disease', 'Symptom_Disease', 'Treatment_Disease',
                'Drug_Disease', 'Anatomy_Disease', 'Frequency_Drug', 
                'Duration_Drug', 'Amount_Drug', 'Method_Drug', 'SideEff-Drug']
    string1 = type1 + '_' + type2
    string2 = type2 + '_' + type1
    string3 = type1 + '-' + type2
    string4 = type2 + '-' + type1
    if (string1 in type_all) :
        return string1
    elif (string2 in type_all) :
        return string2
    elif (string3 in type_all) :
        return string3
    elif (string4 in type_all) :
        return string4
    else:
        return ''

def dist_entity(entity1, entity2):
    '''
        计算两个实体间隔的距离.
        entity1,和entity2为两个字典.
        实体字典的格式为：
            {'T1':{'type': 'Disease', 'pos':[1,2,3]}, 
             'T2':{'type': 'Disease', 'pos':[4,5,6]} }.
        函数返回两个实体间的距离，最左位置和最右位置.
    '''
    pos1 = entity1['pos']
    pos2 = entity2['pos']
    min_val = min(pos1[0], pos2[0])
    max_val = max(pos1[-1], pos2[-1])
    return max_val - min_val + 1, min_val, max_val

def is_in_relation_dict(T1, T2, file_dict):
    '''
        判断两个实体(是字符串) 是否在所给文件的字典中.
        也就是判断这两个实体是否有关系.
        输入字符串T1, T2, 和一个表示标记某个ann文件的字典.
        字典格式为：
            {'text'    : 'abc', 
             'entity'  : {'T1':{'type': 'Disease', 'pos':[1,2,3]}, 
                          'T2':{'type': 'Disease', 'pos':[4,5,6]} },
            'relation' : {'R1':{'type': 'Drug_disease', 'entity':[T1, T2]}},
                          'R2':{'type': 'Drug_disease', 'entity':[T3, T4]}}}
        输出bool类型.
    '''
    file_relation = file_dict['relation']
    for k in file_relation.keys():
        T_list = file_relation[k]['entity']
        if (T1 in T_list) and (T2 in T_list):
            return True;
    return False

def train_dict_to_list(train_file_dict, list_length=150, judge_length_range = (0, 120)):
    '''
        将一个文档的字典(train_file_dict)，转化为一个list。
        list中一个元素，像这样：
        [
            [
                ['对','中','国','成','人','2','型','糖','尿','病','H','B','A','1','C',' ',' ','。','目','标'],
                [-5,  -4,  -3,  -2,  -1,  0,   1,  2,   3,   4,   5,  6,  7, 8,  9,  10, 11,  12,  13, 14],
                [-14, -13, -12, -11,-10, -9, -8,  -7,  -6,  -5,  -4, -3, -2,-1,  0,  1,   2,   3,  4,   5]
            ],
            [0, 1]
        ]
    '''
    train_list = []
    # 得到某个文件的字典
    single_file_dict = train_file_dict
    # 得到这个文件的文本, 字符串类型
    single_file_text = single_file_dict['text']
    # 得到这个文件包含的所有实体
    entity = single_file_dict['entity']
    # 使用列表来存所有实体的名字
    entity_name_list = list(entity.keys()) 
    # 开始两两比较实体的距离，若小于等于judge_length，存入训练集列表
    train_has_relation = []
    train_no_relation = []
    for i in range(len(entity_name_list)):
        for j in range(i + 1, len(entity_name_list)):
            # 两个实体的字符串
            T1 = entity_name_list[i]
            T2 = entity_name_list[j]
            # 得到两个实体见的长度，和左位置，右位置
            dist, left_entity_pos, right_entity_pos = dist_entity(entity[T1], entity[T2])
            # 如果两个实体间的距离小于judge_length
            if (judge_length_range[0] < dist) and (dist <= judge_length_range[1]):
                left_str_pos = left_entity_pos - int((list_length-dist)/2)
                right_str_pos = right_entity_pos + list_length - dist - int((list_length-dist)/2)
                offset = 0
                if (left_str_pos < 0):
                    offset = -left_str_pos
                    left_str_pos = 0
                if (right_str_pos >= len(single_file_text)):
                    right_str_pos = len(single_file_text) - 1
                train_char = [''] * list_length
                for iter, ch in enumerate(single_file_text[left_str_pos:right_str_pos+1]):
                    train_char[iter + offset] = ch
                list_T1 = list(range(-int((list_length-dist)/2), -int((list_length-dist)/2) + list_length))
                list_T2 = list(range(-dist - int((list_length-dist)/2) + 1, list_length - dist - int((list_length-dist)/2) + 1))
                # 判断这两个实体是否有关系
                if (is_in_relation_dict(T1, T2, single_file_dict)):
                    # 两个实体有关
                    train_has_relation.append([[train_char, list_T1, list_T2], [0, 1]])
                else:
                    # 如果这两个实体可能构成潜在的关系
                    if (get_realtion(entity[T1]['type'], entity[T2]['type']) != ''):
                        # 两个实体无关
                        train_no_relation.append([[train_char, list_T1, list_T2], [1, 0]])
                       
    train_list.extend(train_has_relation)
    train_list.extend(train_no_relation)
    return train_list

def val_dict_to_list(val_file_dict, list_length=120, judge_length_range = (100, 200)):
    '''
        将一个文档的字典(val_file_dict)，转化为一个list。
        list中一个元素，像这样：
        [
            [
                ['对','中','国','成','人','2','型','糖','尿','病','H','B','A','1','C',' ',' ','。','目','标'],
                [-5,  -4,  -3,  -2,  -1,  0,   1,  2,   3,   4,   5,  6,  7, 8,  9,  10, 11,  12,  13, 14],
                [-14, -13, -12, -11,-10, -9, -8,  -7,  -6,  -5,  -4, -3, -2,-1,  0,  1,   2,   3,  4,   5]
            ],
            [0, 1]
        ]
    '''
    val_list = []
    # 得到某个文件的字典
    single_file_dict = val_file_dict
    # 得到这个文件的文本, 字符串类型
    single_file_text = single_file_dict['text']
    # 得到这个文件包含的所有实体
    entity = single_file_dict['entity']
    # 使用列表来存所有实体的名字
    entity_name_list = list(entity.keys()) 
    # 开始两两比较实体的距离，若小于等于judge_length，存入训练集列表
    val_has_relation = []
    val_no_relation = []
    for i in range(len(entity_name_list)):
        for j in range(i + 1, len(entity_name_list)):
            # 两个实体的字符串
            T1 = entity_name_list[i]
            T2 = entity_name_list[j]
            # 得到两个实体见的长度，和左位置，右位置
            dist, left_entity_pos, right_entity_pos = dist_entity(entity[T1], entity[T2])
            # 如果两个实体间的距离小于judge_length
            if (judge_length_range[0] < dist) and (dist <= judge_length_range[1]):
                left_str_pos = left_entity_pos - int((list_length-dist)/2)
                right_str_pos = right_entity_pos + list_length - dist - int((list_length-dist)/2)
                offset = 0
                if (left_str_pos < 0):
                    offset = -left_str_pos
                    left_str_pos = 0
                if (right_str_pos >= len(single_file_text)):
                    right_str_pos = len(single_file_text) - 1
                val_char = [''] * list_length
                for iter, ch in enumerate(single_file_text[left_str_pos:right_str_pos+1]):
                    val_char[iter + offset] = ch
                list_T1 = list(range(-int((list_length-dist)/2), -int((list_length-dist)/2) + list_length))
                list_T2 = list(range(-dist - int((list_length-dist)/2) + 1, list_length - dist - int((list_length-dist)/2) + 1))
                # 判断这两个实体是否有关系
                if (is_in_relation_dict(T1, T2, single_file_dict)):
                    # 两个实体有关
                    val_has_relation.append([[val_char, list_T1, list_T2], [0, 1]])
                else:
                    if (get_realtion(entity[T1]['type'], entity[T2]['type']) != ''):
                        # 两个实体无关
                        val_no_relation.append([[val_char, list_T1, list_T2], [1, 0]])
                    
    # 验证数据不抽取，直接计算
    val_list.extend(val_has_relation)
    val_list.extend(val_no_relation)
    return val_list

def test_dict_to_list(test_file_dict, list_length=120, judge_length_range = (100, 200)):
    ''' 
        将一个文档的字典(test_file_dict)，转化为一个list。
            list中一个元素，像这样：
            [
                ['对','中','国','成','人','2','型','糖','尿','病','H','B','A','1','C',' ',' ','。','目','标'],
                [-5,  -4,  -3,  -2,  -1,  0,   1,  2,   3,   4,   5,  6,  7, 8,  9,  10, 11,  12,  13, 14],
                [-14, -13, -12, -11,-10, -9, -8,  -7,  -6,  -5,  -4, -3, -2,-1,  0,  1,   2,   3,  4,   5]
            ]
    '''
    test_list = []
    test_list_tag = []
    # 得到某个文件的字典
    single_file_dict = test_file_dict
    # 得到这个文件的文本, 字符串类型
    single_file_text = single_file_dict['text']
    # 得到这个文件包含的所有实体
    entity = single_file_dict['entity']
    # 使用列表来存所有实体的名字
    entity_name_list = list(entity.keys()) 
    # 开始两两比较实体的距离，若小于等于judge_length，存入训练集列表
    for i in range(len(entity_name_list)):
        for j in range(i + 1, len(entity_name_list)):
            # 两个实体的字符串
            T1 = entity_name_list[i]
            T2 = entity_name_list[j]
            # 得到两个实体见的长度，和左位置，右位置
            dist, left_entity_pos, right_entity_pos = dist_entity(entity[T1], entity[T2])
            # 如果两个实体间的距离小于judge_length
            if (judge_length_range[0] < dist) and (dist <= judge_length_range[1]):
                left_str_pos = left_entity_pos - int((list_length-dist)/2)
                right_str_pos = right_entity_pos + list_length - dist - int((list_length-dist)/2)
                offset = 0
                if (left_str_pos < 0):
                    offset = -left_str_pos
                    left_str_pos = 0
                if (right_str_pos >= len(single_file_text)):
                    right_str_pos = len(single_file_text) - 1
                train_char = [''] * list_length
                for iter, ch in enumerate(single_file_text[left_str_pos:right_str_pos+1]):
                    train_char[iter + offset] = ch
                list_T1 = list(range(-int((list_length-dist)/2), -int((list_length-dist)/2) + list_length))
                list_T2 = list(range(-dist - int((list_length-dist)/2) + 1, list_length - dist - int((list_length-dist)/2) + 1))
                # 如果判断这两个实体有潜在关系
                if (get_realtion(entity[T1]['type'], entity[T2]['type']) != ''):
                    test_list.append([train_char, list_T1, list_T2])
                    test_list_tag.append([T1, T2])
    return test_list, test_list_tag

print('正在处理训练集数据...')
i = 0
train_list = []
for key in train_dict.keys():
    lis = train_dict_to_list(train_dict[key], 150, (0, 120))
    train_list.extend(lis)
    i = i+1
    print(i, end = ' ')
print('\n')
random.shuffle(train_list)
print('整个数据集长度为', len(train_list)) 
with open(os.path.join(data_dict_path, 'train_list_150_0-120.plk'), 'wb') as f:
    pickle.dump(train_list, f)
print('训练样本处理完毕！')
