# 2018 天池大数据竞赛：人工智能辅助构建知识图谱

## 第一赛季：命名实体识别

第一赛季问题是命名实体识别问题，我们才用的是 lstm-crf 模型，rawdata 文件夹中放了训练与测试数据，code 文件夹中放了我们的代码，代码主要包括预处理和模型。看一下就基本知道了。

## 第二赛季：关系抽取

第二赛季我们团队使用的是一个比较简单的卷积神经网络，用于对实体之间的关系进行分类。

运行代码可执行如下步骤：

数据预处理

```shell
cd ./second_season/code
tar xf rawdata.zip
python data_raw_to_dict.py
python dict_to_list.py
python list_to_model.py
```

训练模型

```shell
python model_for_0-120.py
```
