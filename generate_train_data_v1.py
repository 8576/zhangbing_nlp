import re
import os
import gensim
from gensim import corpora
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

    def convert_to_token(self, string):
        for column in self.operate_column:
            string[column] = string[column].split(',')
        return string

    def convert_to_id(self, string):
        #for column in self.operate_column:
        string[self.column] = self.dictionary.doc2idx(string[self.column])
        return string

    def gen_dataframe(self, tokenfiles):
        # tokenfile = r'D:\BaiduNetdiskDownload\data\ieee_zhihu_cup\question_train_set3.txt'
        flag = False
        for f in tokenfiles:
            if not flag:
                data = pd.read_csv(f, sep='\t', header=0)
                flag = True
            else:
                data = pd.merge(left=data, right=pd.read_csv(f, sep='\t', header=0),
                                on='question_id', how='left')
                # left_on = 'question_id', right_on = 'question_id',
        data = data.dropna(axis=0, how='any')
        # print(data[:3])
        self.dataframe = data.apply(self.convert_to_token, axis=1)
        # print(self.dataframe[:3])
        # print()
        # 释放内存
        # del data
    def vocabulary(self):
        vocab = dict()
        for colum in self.operate_column:
            if re.findall('label|topic', colum):
                corps = list(self.dataframe[colum])
                dictionary = gensim.corpora.Dictionary(corps)
                vocabulary_token2id = dictionary.token2id
                vocabulary_id2token = {v: k for k, v in vocabulary_token2id.items()}
                vocab[self.operate_column[1]] = [vocabulary_token2id, vocabulary_id2token, dictionary]
            else:
                corps = list(self.dataframe[colum])
                corps.extend([['_PAD', '_UNK']])
                dictionary = gensim.corpora.Dictionary(corps)
                vocabulary_token2id = dictionary.token2id
                vocabulary_id2token = {v: k for k, v in vocabulary_token2id.items()}
                vocab[self.operate_column[0]] = [vocabulary_token2id, vocabulary_id2token, dictionary]
        return vocab

    def train_values(self):
        vocab = self.vocabulary()
        for colum in self.operate_column:
            self.column = colum
            self.dictionary = vocab.get(colum)[2]
            self.dataframe = self.dataframe.apply(self.convert_to_id, axis=1)
        self.class_nums = len(vocab.get(self.operate_column[1])[0])

        # labels = self.one_hot_lable(class_nums)
        del_columns = list(set(self.dataframe.columns) - set(self.operate_column))
        # print('columns:', self.dataframe.columns)
        # print('del_columns', del_columns, type(del_columns))
        for dc in del_columns:
            del self.dataframe[dc]
        del vocab[self.operate_column[0]][1:]
        del vocab[self.operate_column[1]][1:]
        self.dataframe = self.dataframe.apply(self.one_hot_lable, axis=1)
        x = self.dataframe[self.operate_column[0]]
        labels = self.dataframe[self.operate_column[1]]
        # self.dataframe[self.operate_column[0]] = pad_sequences(self.dataframe[self.operate_column[0]],
        #                                                        maxlen=200, value=0)
        return x, labels

    def one_hot_lable(self, string):
        line_label = np.zeros(shape=(self.class_nums,), dtype=np.int)
        line_label[string[self.operate_column[1]]] = 1
        string[self.operate_column[1]] = line_label
        return string




if __name__ == '__main__':
    cache_path = os.path.dirname(os.path.abspath('__file__')) + '\data'
    print(cache_path)
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    hdffile = cache_path + '\data.hdf'
    pickfile = cache_path + '\data.pickle'
    tempfile = cache_path + '\label.pickle'
    columns = ['desc_word', 'topic_ids']
    dataobj = GenData(operate_column=columns)
    tokenfiles = [r'D:\BaiduNetdiskDownload\data\ieee_zhihu_cup\question_train_set3.txt',
                  r'D:\BaiduNetdiskDownload\data\ieee_zhihu_cup\question_topic_train_set3.txt']
    dataobj.gen_dataframe(tokenfiles)
    print('stage one end.....')
    vocab = dataobj.vocabulary()
    print('stage one two.....')
    X, labels = dataobj.train_values()
    print('stage one third.....')
    X = pad_sequences(X, maxlen=200, value=0)
    # print(X[: 10])
    # print('-' * 100)
    # print(Y[:10])
    # print('x , y shape', X.shape, Y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        labels,
                                                        test_size=0.1)
    del dataobj
    print('x_train, x_test, y_train, y_test', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    with open(tempfile, 'wb') as temf:
        pickle.dump({'y_train': y_train, 'y_test': y_test})

    with open(pickfile, 'wb') as pf:
        tokens2id = {'vocabulary_x_token2id': vocab.get(columns[0])[0],
                     'vocabulary_y_token2id': vocab.get(columns[1])[0]}
        pickle.dump(tokens2id, pf)
    print('not to use tolist!')
    with h5py.File(hdffile, 'w') as f:
        f['x_train'], f['x_test'], f['y_train'], f['y_test'] = x_train, x_test, y_train.tolist(), y_test.tolist()
        # , f['y_train'], f['y_test']  , y_train, y_test



