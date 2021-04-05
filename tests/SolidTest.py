#!/usr/bin/env python
import codecs
d_ = lambda _:codecs.latin_1_decode(_)[0]  

import numpy as np

class Solid(object):
    def __init__(self, a, i):
        b = a.tobytes()
        self.i = i 
        self.label = d_( b[16*i:16*i+4] )
        self.numPrim = a[i,1]
        self.primOffset = a[i,2]
        self.extent = a[i,3].view(np.float32)
    def __repr__(self):
        return "Solid(%d) %10s numPrim:%3d primOffset:%4d extent:%10.4f " % ( self.i, self.label, self.numPrim, self.primOffset, self.extent )


if __name__ == '__main__':
    a = np.load("/tmp/SolidTest.npy")
    for i in range(len(a)):
        so = Solid(a,i)
        print(so)
    pass


