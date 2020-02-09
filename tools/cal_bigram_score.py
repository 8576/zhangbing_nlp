#! -*- coding:utf-8 -*-
import pickle


class CalScore(object):

    def __init__(self):
        with open(r'D:\pro\Build_bigram\data\bigram.pickle', mode='rb') as f:
            self.bigramdict = pickle.load(f)

    def cal(self, vc_list):
        cal_res = dict()
        for f, front in enumerate(vc_list):
            if front in self.bigramdict:
                for t, tail in enumerate(vc_list):
                    if t == f:
                        continue
                    else:
                        if tail in self.bigramdict[front]:
                            cal_res[str(front) + ',' + str(tail)]\
                                = int(self.bigramdict[front][tail])
                        elif tail[-2:] in self.bigramdict[front]:
                            cal_res[str(front) + ',' + str(tail)] \
                                = int(self.bigramdict[front][tail[-2:]])
            elif front[-2:] in self.bigramdict:
                for t, tail in enumerate(test_data):
                    if t == f:
                        continue
                    else:
                        if tail in self.bigramdict[front[-2:]]:
                            cal_res[str(front) + ',' + str(tail)]\
                                = int(self.bigramdict[front[-2:]][tail])
                        elif tail[-2:] in self.bigramdict[front[-2:]]:
                            cal_res[str(front) + ',' + str(tail)] \
                                = int(self.bigramdict[front[-2:]][tail[-2:]])
        res_list = sorted(cal_res.items(), key=lambda x: x[1], reverse=True)
        print(res_list)
        return res_list


if __name__ == '__main__':
    test_data = ['中央军委', '主席', '习近平', '近日', '对', '军队', '做好', '新型', '冠状病毒', '感染', '的', '肺炎', '疫情', '防控', '工作', '作出',
                 '重要', '指示', '全军', '要', '在', '党中央', '和', '中央军委', '统一指挥', '下', '牢记', '人民军队', '宗旨', '闻令', '而动', '勇挑重担',
                 '敢', '打硬仗', '积极', '支援', '地方', '疫情', '防控']
    calscore = CalScore()
    calscore.cal(test_data)

