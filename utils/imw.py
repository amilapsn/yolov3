"""This script is to get the weight for the each training image s.t.
the weight represent how important the image class-wise.
These calculated weights is used for the `sampler`
"""

import csv
import numpy as np

def f(train):
    ims = open(train).readlines()
    txts = [im.replace('jpg','txt').strip().replace('images','labels') for im in ims]
    arr = []
    p = {}

    for i,txt in enumerate(txts):
        objects = open(txt).readlines()
        objects = [int(obj.strip().split()[0]) for obj in objects]
        pp = {}
        for obj in objects:
            if obj in p:
                p[obj] += 1
            else:
                p[obj] = 1
            if obj in pp:
                pp[obj] += 1
            else:
                pp[obj] = 1
        arr.append(pp)

    n = len(p)
    m = len(arr)

    MAT = np.zeros((m,n))

    for i,d in enumerate(arr):
        for key in d:
            MAT[i,key-1] += d[key]

    p = MAT.sum(axis=0)
    p_MAT = np.repeat([list(p)],m,axis=0)

    MAT = MAT * p_MAT

    M = MAT.sum(axis=1)
    M[M==0] = np.sort(M)[-1]
    ret = MAT.sum()/(M)
    return ret

if __name__ == '__main__':
    f()
