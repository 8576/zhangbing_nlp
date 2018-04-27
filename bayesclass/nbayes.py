# -*- coding:utf-8 -*-
import os
import sys
import numpy as np


class Nbayes(object):
    def __init__(self):
        self.vocabulary = []
        self.idf = 0
        self.tf = 0
        self.tdm = 0
        self.Pcates = []
        self.vocablen = 0
        self.doclength = 0
        self.testset = 0

    def train_set(self, trainset, classVec):
        self.cate_prob(classVec)
        self.doclength = len(trainset)
        tempset = set()
        [tempset.add(word) for doc in trainset for word in doc]
        self.vocabulary = list(tempset)
        self.calc_wordfreq(trainset)
        self.build_tdm()

    def cate_prob(self, classVec):
        self.labels = classVec
        labeltemps = set(self.labels)
        for labeltemp in labeltemps:
            self.Pcates[labeltemp] = float(self.labels.count(labeltemp)) / float(len(self.labels))

    def calc_wordfreq(self, trainset):
        self.idf = np.zeros([1, self.vocablen])
        self.tf = np.zeros([self.doclength, self.vocablen])

        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx, self.vocabulary.index(word)] += 1
            for signleword in set(trainset[indx]):
                self.idf[0, self.vocabulary.index(signleword)] += 1

    def build_tdm(self):
        self.tdm = np.zeros([len(self.Pcates), self.vocablen])
        sumlist = np.zeros([len(self.Pcates), 1])
        for indx in range(self.doclength):
            self.tdm[self.labels[indx]] += self.tf[indx]

            sumlist[self.labels[indx]] = np.sum(self.tdm[self.labels[indx]])
        self.tdm = self.tdm / sumlist

    def map2vocab(self, testdata):
        self.testset = np.zeros([1, self.vocablen])
        for word in testdata:
            self.testset[0, self.vocabulary.index(word)] += 1

    def predict(self, testset):
        if np.shape(testset)[1] != self.vocablen:
            print('输入错误！')
            exit(0)
        prevalue = 0
        preclass =''
        for tdm_vect, keyclass in zip(self.tdm, self.Pcates):
            temp = np.sum(testset * tdm_vect * self.Pcates[keyclass])
            if temp > prevalue:
                prevalue = temp
                preclass = keyclass
        return preclass
    def calc_tfidf(self, trainset):
        self.idf = np.zeros([1, self.vocablen])
        self.tf = np.zeros([self.doclength, self.vocablen])
        for indx in range(self.doclength):
            for word in trainset[indx]:
                self.tf[indx, self.vocabulary.index(word)] += 1
            self.tf[indx] = self.tf[indx] / float(len(trainset[indx]))
            for signleword in set(trainset[indx]):
                self.idf[0, self.vocabulary.index((signleword))] += 1
        self.idf = np.log(float(self.doclength)/ self.idf)
        self.tf = np.multiply(self.tf, self.idf)


if __name__ == '__main__':

    dataSet, listClass = loadDataSet() # 句子向量，所属类别
    nb = Nbayes()
    nb.train_set(dataSet, listClass)
    nb.map2vocab(dataSet[0])
    print(nb.predict(nb.testset))
