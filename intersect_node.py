#!/usr/bin/env python
import os
import numpy as np

from glob import glob
import matplotlib.pyplot as plt 

class IntersectNodeTest(object):
    def __init__(self, path):
        a = np.load(path) 

        x = a[:,0]
        y = a[:,1]
        z = a[:,2]
        t = a[:,3]

        self.path = path 
        self.a = a
        self.x = x
        self.y = y
        self.z = z
        self.t = t

if __name__ == '__main__':

    paths="/tmp/intersect_node_tests/circle_scan/*.npy" 
    paths = sorted(glob(paths))

    num = len(paths)
    fig, axs = plt.subplots(num)
    for i in range(num):
        path = paths[i]
        tst = IntersectNodeTest(path)
        ax = axs[i]
        ax.plot( tst.x, tst.z )
    pass
    fig.show()


