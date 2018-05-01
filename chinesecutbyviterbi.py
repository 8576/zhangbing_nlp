# -*- coding:utf-8 -*-
import numpy as np
import math
import codecs
import jieba
import random
import os
import re





infinite = float(-2 ** 31)
# print(data)

def log_normalize(a):
    # print('a=\n', a)
    if a:
        s = np.sum(a)
        # print('if s =\n', s)
    if s == 0:
        s += math.pow(2, -20)
        # print('s=0\n')
    s = math.log(s)
    for i in range(len(a)):
        # print('a[{}] =\n'.format(i), a[i])
        if a[i] == 0:
            a[i] = infinite
            # pass
        else:
            a[i] = math.log(a[i]) - s



def mle(n): #B0/M1/E2/S3
    N = 4
    M = 65536
    last_q = 2
    pi = [0] * N
    a = [[0] * N for x in range(N)]
    b = [[0] * M for y in range(N)]

    # with codecs.open('/home/wall/xxml/lectures/24.HMM/pku_training.utf8', encoding='utf-8') as f:
    #     data = f.read().split('  ')
    data = getnewdata(n=n)
    for k, token in enumerate(data):
        token = token.strip()
        # print(token)
        n = len(token)
        if n <= 0:
            continue
        if n == 1:
            pi[3] += 1
            a[last_q][3] += 1
            b[3][ord(token[0])] += 1
            last_q = 3
            continue
        pi[0] += 1
        pi[2] += 1
        pi[1] += (n - 2)

        a[last_q][0] += 1
        last_q = 0
        if n == 2:
            a[0][2] += 1
        else:
            a[0][1] += 1
            a[1][1] += (n - 3)
            a[1][2] += 1
        b[0][ord(token[0])] += 1
        b[2][ord(token[-1])] += 1
        for item in token[1:-1]:
            b[1][ord(item)] += 1
    # print('before normal'.center(100, '*'))
    # print('pi = \n', pi)
    # print('a = \n', a)
    # print('b = \n', b)
    # print('end normal'.center(100, '*'))

    log_normalize(pi)
    for time in range(N):
        log_normalize(b[time])
        log_normalize(a[time])
    return pi, a, b


def list_write(filename, listobj):
    data = np.array(listobj)
    print(data.shape)
    if len(data.shape) == 1:
        # print('write..=\n', data)
        with open(filename, 'w') as f:
            f.write('  '.join(map(str, data)))
    elif len(data.shape) == 2:
        with open(filename, 'w') as f:
            for index in data:
                f.write('  '.join(map(str, index)))
                f.write('\n')

def viterbi(pi, A, B, o):
    T = len(o)
    delta = [[0 for item in range(4)] for time in range(T)]
    pre = [[0 for item in range(4)] for time in range(T)]
    # 计算delta(1)
    q = 0
    for i in range(4):
        delta[0][i] = pi[i] + B[0][ord(o[0])]
    for t in range(1, T):
        for i in range(4):
            delta[t][i] = delta[t-1][0] + A[0][i]
            for j in range(1, 4):
                vj = delta[t-1][j] + A[j][i]
                if delta[t][i] < vj:
                    delta[t][i] = vj
                    # q = j
                    pre[t][i] = j
            delta[t][i] += B[i][ord(o[t])]

    decode = [-1 for item in range(T)]
    for index in range(1, 4):
        if delta[T-1][index] > delta[T-1][index -1]:
            q = index
    decode[T-1] = q
    for i in range(T-2, -1, -1):
        q = pre[i+1][q]
        decode[i] = q

    return decode


def cut(strings, decode):
    N = len(strings)
    i = 0
    while i < N:
        if decode[i] == 0 or decode[i] == 1:
            j = i
            while True:
                j += 1
                if j == N or decode[j] == 2:
                    break
            print(strings[i:j+1], end=' | ')
            i = j + 1
        elif decode[i] == 3:
            print(strings[i:i + 1], end=' | ')
            i += 1
        else:
            print('Error:', i, decode[i])
            i += 1


def getnewdata(n):
    path = '../yuliao/news_data'
    # print(os.listdir(path))
    cutstring = ''
    for artic in random.choices(os.listdir(path), k=n):
        with open('/'.join([path, artic])) as f:
            cutstring = ''.join([cutstring, f.read()])
    print('{}篇原始语料加载完毕...'.format(n))
    # print(cutstring)
    # cutstring = re.sub('\n+', '', cutstring, flags=re.DOTALL)
    print('jieba切词完毕...')
    jieba.enable_parallel(5)
    data = jieba.cut(cutstring)
    # print(list(data))
    # print('$' * 100)
    return data


if __name__ == '__main__':
    # 用100篇文档作为结巴切词的训练数据
    pi, a, b = mle(600)
    # 把pi, a, b写入到文件中取
    # list_write('pi.txt', pi)
    # list_write('a.txt', a)
    # list_write('b.txt', b)

    cutstring = ''
    path = '../yuliao/news_data'

    for artic in random.choices(os.listdir(path), k=1):
        with open('/'.join([path, artic])) as f:
            cutstring = ''.join([cutstring, f.read()])
    print('@' * 50, cutstring, '@' * 50, sep='\n')
    decode = viterbi(pi, a, b, cutstring)
    cut(cutstring, decode)





















