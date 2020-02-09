#! -*- coding:utf-8 -*-
import pickle
import re
from pyhanlp import JClass


class Process(object):
    def __init__(self):
        self.condition_data = dict()

    def process(self, seg_list):
        for i in range(1, len(seg_list)):
            bigram = seg_list[i - 1] + '@' + seg_list[i]
            if bigram in self.condition_data:
                self.condition_data[bigram] += 1
            else:
                self.condition_data[bigram] = 1

    def write(self, path):
        with open(path, 'w', encoding='utf8') as f:
            for key, frequency in sorted(self.condition_data.items(), key=lambda x: x[1], reverse=True):
                string = key + ' ' + str(frequency) + '\n'
                f.write(string)


if __name__ == '__main__':
    p = Process()
    NShortSegment = JClass("com.hankcs.hanlp.seg.NShort.NShortSegment")
    nshort_segment = NShortSegment().enableCustomDictionary(False).enablePlaceRecognize(
        True).enableOrganizationRecognize(True)
    with open('data/test_data', encoding='utf8') as f:
        for line in f:
            line = re.sub('\s', '', line)
            if line:
                res = nshort_segment.seg(line)
                res = list(map(lambda x: re.sub('/.*', '', str(x)), res))
                print(res)
                p.process(res)
    p.write('./data/self.ngram.txt')

