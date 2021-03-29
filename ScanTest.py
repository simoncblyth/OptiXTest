#!/usr/bin/env python
"""

"""
import os
import numpy as np
from glob import glob

try:
    import matplotlib.pyplot as plt 
except ImportError:
    plt = None
pass

try:
    import pyvista as pv
except ImportError:
    pv = None
pass

def plot3d(pos, grid=False):
    pl = pv.Plotter()
    pl.add_points(pos, color='#FFFFFF', point_size=2.0 )  
    if grid:
        pl.show_grid()
    pass
    cp = pl.show()
    return cp


class ScanTest(object):
    def __init__(self, path):
        a = np.load(path) 

        ori = a[:,0]
        dir = a[:,1]
        post = a[:,2]
        isect = a[:,0,3].view(np.int32)  

        tot = len(a)
        hit = np.count_nonzero( isect == 1 )  
        miss = np.count_nonzero( isect == 0 )  

        self.path = path 
        self.a = a
        self.ori = ori
        self.dir = dir
        self.post = post
        self.isect = isect
        self.tot = tot 
        self.hit = hit 
        self.miss = miss

    def __repr__(self):
        return "%s : tot %d hit %d miss %d " % (self.path, self.tot, self.hit, self.miss)
 

if __name__ == '__main__':


    #solid = "sphere"
    #solid = "zsphere"
    #solid = "cone"
    #solid = "convexpolyhedron_cube" 
    #solid = "convexpolyhedron_tetrahedron" 
    solid = "hyperboloid"
    #solid = "box3"
    #solid = "plane"
    #solid = "slab"
    #solid = "cylinder"
    #solid = "disc"

    #scan = "circle"
    scan = "rectangle"

    base = "/tmp/ScanTest_scans"
    path = "%(base)s/%(scan)s_scan/%(solid)s.npy" % locals()

    fig, axs = plt.subplots(1)
    if not type(axs) is np.ndarray: axs = [axs] 

    st = ScanTest(path)
    print(st)

    ax = axs[0]
    ax.set_aspect('equal')
    ax.scatter( st.post[:,0], st.post[:,2], s=0.1 )
    ax.scatter( st.ori[:,0],  st.ori[:,2] )
    scale = 10.
    ax.scatter( st.ori[:,0] + st.dir[:,0]*scale, st.ori[:,2]+st.dir[:,2]*scale )
    fig.show()

    plot3d( st.post[:,:3] )

