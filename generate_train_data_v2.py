# -*- coding: utf-8 -*-
"""
Author: zhangbing
Date: 2018.12.14
The scritp is used pre-processing data of nlp tasks.
    1. Vocabulary is needed. Generally, it's a dict type, so-called token2id
    2. Convert corporas to numbers according to vocabulary
    3. Featue of every item  must be same length. "pad_sequence" may be usefull
    4. Label is a sparse vector like one-hot, or multi-one with multi-labels
    5. All words, pandas is the-state-art in pre-processing
"""

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
import re
import os
import gensim
import pickle
import pandas as pd
import numpy as np
import h5py
from tflearn.data_utils import  pad_sequences
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)


class GenData(object):
    def __init__(self, operate_column):
        self.operate_column = operate_column
        # 每个文件中存放数据条数
        self.file_per_data = 500000
        # 用于标记文件存储
        self.count = 0
        self.x = []
        self.y = []
        self.dirname = os.path.dirname(os.path.abspath('__file__')) + '\data'

    def convert_to_token(self, string):
        """
        每个句子中的词是由逗号分隔开的
        :param string:
        :return:
        """
        for column in self.operate_column:
            string[column] = string[column].split(',')
        return string

    def convert_to_id(self, string):
        """
        借用了gensim中doc2idx这个接口，直接把一个sentence的token转成数值
        :param string:
        :return:
        """
        string[self.column] = self.dictionary.doc2idx(string[self.column])
        return string

    def gen_dataframe(self, tokenfiles):
        """
        把两个dataframe合并
        :param tokenfiles:
        :return:
        """
        flag = False
        for f in tokenfiles:
            if not flag:
                self.dataframe = pd.read_csv(f, sep='\t', header=0)
                flag = True
            else:
                # 相当于左连接
                # 如果连接的key不一致，就用left_on 和 right_on连接；如果一致就用on
                self.dataframe = pd.merge(left=self.dataframe, right=pd.read_csv(f, sep='\t', header=0),
                                        on='question_id', how='left')
                # left_on = 'question_id', right_on = 'question_id',

        # print(data[:3])
        del_columns = list(set(self.dataframe.columns) - set(self.operate_column))
        # 删除多余的列，节约内存
        self.dataframe.drop(axis=1, columns=del_columns, inplace=True)
        # for dc in del_columns:
        #     del self.dataframe[dc]
        self.dataframe = self.dataframe.dropna(axis=0, how='any')
        # axis=1，是按行来传入，处理的时候是按行来处理
        # axis-0, 是按列来传入，处理的时候是按列来处理
        self.dataframe = self.dataframe.apply(self.convert_to_token, axis=1)
        # print(self.dataframe[:3])
        # print()
        # 释放内存
        # del data
    def vocabulary(self):
        vocab = dict()
        for colum in self.operate_column:
            if re.findall('label|topic', colum):
                corps = list(self.dataframe[colum])
                # 调用gensim接口很方便的生成词汇表
                # dictionary.token2id 是按词频来生成的
                # dictonary对象有个token2id属性
                # dictonary对象有个doc2idx方法，用于将文档数值化
                dictionary = gensim.corpora.Dictionary(corps)
                vocabulary_token2id = dictionary.token2id
                vocabulary_id2token = {v: k for k, v in vocabulary_token2id.items()}
                vocab[self.operate_column[1]] = [vocabulary_token2id, vocabulary_id2token, dictionary]
            else:
                corps = list(self.dataframe[colum])
                # 如果是特征，则需要考虑pad和unk
                corps.extend([['_PAD', '_UNK']])
                dictionary = gensim.corpora.Dictionary(corps)
                vocabulary_token2id = dictionary.token2id
                vocabulary_id2token = {v: k for k, v in vocabulary_token2id.items()}
                vocab[self.operate_column[0]] = [vocabulary_token2id, vocabulary_id2token, dictionary]
        return vocab

    def train_values(self, vocab):
        """
        :param vocab:
        :return:
        """
        for colum in self.operate_column:
            self.column = colum
            self.dictionary = vocab.get(colum)[2]
            self.dataframe = self.dataframe.apply(self.convert_to_id, axis=1)
        self.class_nums = len(vocab.get(self.operate_column[1])[0])

        # labels = self.one_hot_lable(class_nums)
        # del_columns = list(set(self.dataframe.columns) - set(self.operate_column))
        # # print('columns:', self.dataframe.columns)
        # # print('del_columns', del_columns, type(del_columns))
        # for dc in del_columns:
        #     del self.dataframe[dc]

        # 删除vocabulary中多余的条目，以释放内存
        del vocab[self.operate_column[0]][1:]
        del vocab[self.operate_column[1]][1:]
        # self.dataframe = self.dataframe.apply(self.one_hot_lable, axis=1)
        # x = self.dataframe[self.operate_column[0]]
        # labels = self.dataframe[self.operate_column[1]]
        # self.dataframe[self.operate_column[0]] = pad_sequences(self.dataframe[self.operate_column[0]],
        # 不放回的row随机抽样，相当于shuffle操作
        # reset_index 不维护原来的索引
        # axis = 0 抽row，默认是0
        self.dataframe = self.dataframe.sample(frac=1, replace=False).reset_index(drop=True)
        return None

    def one_hot_lable(self, string):
        line_label = np.zeros(shape=(self.class_nums,), dtype=np.int)
        line_label[string[self.operate_column[1]]] = 1
        string[self.operate_column[1]] = line_label
        return string
    def calculate_file_index(self):
        """
        计算将原数据，划分成几个批次
        :return:
        """
        datafram_length = len(self.dataframe)
        print('dataframe length: {}'.format(datafram_length))
        div = datafram_length // self.file_per_data
        if div == 0:
            self.file_index = [datafram_length -1]
        else:
            self.file_index = [self.file_per_data * index - 1 for index in range(1, div + 1)]
        if datafram_length - self.file_index[-1] < 1000:
            self.file_index[-1] = datafram_length -1
        else:
            self.file_index.append(datafram_length -1)
        print('file index', self.file_index)


    def write_pices(self, string):
        """
        把原数值sentence 做成定长，把label处理成one-hot后，维度很高，内存爆炸
        考虑：处理一个批次然后存入文件，然后释放内存。再接着处理。
        :param string:
        :return:
        """
        line_label = np.zeros(shape=(self.class_nums,), dtype=np.int)
        line_label[string[self.operate_column[1]]] = 1
        # print(string[self.operate_column[0]])
        # 出入的是一个类似序列的对象,像这样[[]]
        tx = pad_sequences([string[self.operate_column[0]]], maxlen=200, value=0)
        # print(tx)
        ty = line_label.tolist()
        self.x.append(tx[0])
        self.y.append(ty)
        if self.count in self.file_index:
            # h5py在dataset写入的时候，长度需一致，否则会报错。
            with h5py.File(os.path.join(self.dirname, 'traindta_' + str(self.count) + '.hdf5'), 'w') as f:
                f['x_train'] = self.x
                f['y_train'] = self.y
            self.x.clear()
            self.y.clear()
            print('{} had been finished!'.format(self.count))
        self.count += 1
        return ''


def xdata():
    cache_path = os.path.dirname(os.path.abspath('__file__')) + '\data'
    print(cache_path)
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    # hdffile = cache_path + '\data.hdf'
    pickfile = cache_path + '\labels.pickle'
    # tempfile = cache_path + r'\tmp.pickle'
    columns = ['desc_word', 'topic_ids']
    dataobj = GenData(operate_column=columns)
    tokenfiles = [r'D:\BaiduNetdiskDownload\data\ieee_zhihu_cup\question_train_set3.txt',
                  r'D:\BaiduNetdiskDownload\data\ieee_zhihu_cup\question_topic_train_set3.txt']
    dataobj.gen_dataframe(tokenfiles)
    print('stage one end.....')
    vocab = dataobj.vocabulary()
    print('stage one two.....')
    print('未修改前的vocab ', len(vocab.get('topic_ids')))
    dataobj.train_values(vocab=vocab)
    print('修改后的vocab ', len(vocab.get('topic_ids')))
    print('stage one third.....')
    dataobj.calculate_file_index()
    dataobj.dataframe.apply(dataobj.write_pices, axis=1)
    print('train data fineshed!')



    # # del dataobj
    # print('x_train, x_test, y_train, y_test', x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    #
    # with open(tempfile, 'wb') as temf:
    #     pickle.dump({'y_train': y_train, 'y_test': y_test}, temf)
    #
    with open(pickfile, 'wb') as pf:
        tokens2id = {'vocabulary_x_token2id': vocab.get(columns[0])[0],
                     'vocabulary_y_token2id': vocab.get(columns[1])[0]}
        pickle.dump(tokens2id, pf)
    # print('not to use tolist!')
    # with h5py.File(hdffile, 'w') as f:
    #     f['x_train'], f['x_test'] = x_train, x_test
        # , f['y_train'], f['y_test']  , y_train, y_test

def ydata():
    cache_path = os.path.dirname(os.path.abspath('__file__')) + '\data'
    hdffile = cache_path + '\data.hdf'
    pickfile = cache_path + '\labels.pickle'
    tempfile = cache_path + r'\tmp.pickle'
    with open(pickfile, 'rb') as vocabf:
        tokens2id = pickle.load(vocabf)
        class_nums = len(tokens2id.get('vocabulary_y_token2id'))
    del tokens2id

    with open(tempfile, 'rb') as f:
        data = pickle.load(f)
        y_train, y_test = data.get('y_train'), data.get('y_test')
    del data
    with h5py.File(hdffile, mode='r+') as hd:

        for i, f in enumerate([y_train, y_test]):

            if i == 0:
                key = 'y_train'
            else:
                key = 'y_test'
            hd[key] = []
            for line in f:
                line_label = np.zeros(shape=(class_nums,), dtype=np.int)
                line_label[line] = 1
                hd[key].append(line_label.tolist())



if __name__ == '__main__':
    xdata()
    # ydata()



