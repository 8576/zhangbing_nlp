# -*- coding:utf-8 -*-
import numpy as np
from collections import OrderedDict

# from numpy import *
# sunny, cloudy, rainly
transp = np.array([[.5, .375, .125],
                  [.25, .125, .625],
                  [.25, .375, .375]])
# dry, dryish, soggy
obs = np.array([[.6, .2, .05],
               [.25, .25, .25],
               [.05, .10, .50]])
pi = np.array([.63, .17, .20])

# 观测序列(干旱， 干燥，潮湿)
# 直接计算方法
def comdecode(pi, transp, obs):
    print('直接计算法：')
    stateone = pi * obs[:, 0].T
    print(stateone, stateone.argmax())

    statetwo = stateone.dot(transp) * obs[:, 1]
    print(statetwo, statetwo.argmax())

    statethree = statetwo.dot(transp) * obs[:, 2]
    print(statethree, statethree.argmax())

# viterbi 算法
def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    :param obs: 观测
    :param states: 隐状态
    :param start_p: 初始状态
    :param trans_p: 转移概率
    :param emit_p: 发射概率 （隐状态转换为显示是序列的概率）
    :return:
    """
    print('Viterbi 算法计算：\n')
    V = [{}] # V[时间][隐状态] = 概率
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
    for t in range(1, len(obs)):
        V.append({})
        for y in states:
            V[t][y] = np.max([V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]] for y0 in states])
    result = []
    for vector in V:
        temp = {}
        # print(argmax(list(vector.values())))
        # print(vector.keys())
        temp[max(vector, key=vector.get)] = np.max(vector.values())
        result.append(temp)
    return result


if __name__ == '__main__':
    states = ['Sunny', 'Cloudy', 'Rainy']
    obs = ('dry', 'dryish', 'soggy')
    start_p = {'Sunny': .63, 'Cloudy': .17, 'Rainy': .20}
    trans_p = {
        'Sunny': {'Sunny': .5, 'Cloudy': .375, 'Rainy': .2},
        'Cloudy': {'Sunny': .25, 'Cloudy': .125, 'Rainy': .625},
        'Rainy': {'Sunny': .25, 'Cloudy': .375, 'Rainy': .375}}

    emit_p = {
        'Sunny': {'dry': .6, 'dryish': .2, 'soggy': .05},
        'Cloudy': {'dry': .25, 'dryish': .25, 'soggy': .25},
        'Rainy': {'dry': .05, 'dryish': .1, 'soggy': .5}
    }

    print(viterbi(obs, states, start_p, trans_p, emit_p))


