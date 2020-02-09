#! -*- coding:utf-8 -*-
import os
import re
import pickle


class BigramtoHashTable(object):
    def __init__(self):
        self.table = dict()
        self.pattern_split = re.compile('@| ')


    def process(self, files):
        """
        :param files: bigram file list
        :return:
        """
        for rf in [f for f in files if os.path.exists(f)]:
            with open(rf, mode='r', encoding='utf-8') as bf:
                for line in bf:
                    line = line.strip()
                    if line:
                        given, y, frequency = self.pattern_split.split(line)
                        if given in self.table:
                            if y in self.table[given]:
                                self.table[given][y] += frequency
                            else:
                                self.table[given][y] = frequency
                        else:
                            self.table[given] = {y: frequency}
        print('hashtable 中共计{}条数据'.format(self.count()))


    def write(self, to_path):
        if not os.path.exists(os.path.dirname(to_path)):
            os.mkdir(os.path.dirname(to_path))
        with open(to_path, mode='wb') as f:
            pickle.dump(self.table, f)
        self.count()
        print('bigram map 持久化完毕！\n 持久化路径：{} \n 总共持久化{}条数据'.format(to_path, self.count()))

    def count(self):
        term_size = 0
        for d in self.table.values():
            term_size += len(d)
        return term_size


if __name__ == '__main__':
    bigram = BigramtoHashTable()
    files = [r'D:\pro\Build_bigram\data\CoreNatureDictionary.ngram.txt']
    bigram.process(files)
    bigram.write(r'D:\pro\Build_bigram\data\bigram.pickle')