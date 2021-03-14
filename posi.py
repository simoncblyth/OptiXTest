#!/usr/bin/env python

import os, glob, logging
log = logging.getLogger(__name__)

import numpy as np
np.set_printoptions(suppress=True)

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

def plot3d(pos):
    pl = pv.Plotter()
    pl.add_points(pos, color='#FFFFFF', point_size=2.0 )  
    pl.show_grid()
    cp = pl.show()
    return cp

def plot2d(img):
    """
    plot2d(pxid)
    plot2d(bindex)
    """
    fig, axs = plt.subplots(1)
    axs.imshow(img) 
    fig.show()



dir_ = lambda:os.environ.get("OUTDIR", os.getcwd())
path_ = lambda name:os.path.join(dir_(), name)
load_ = lambda name:np.load(path_(name))

def sdf_sphere(p,sz):
    """ 
    :param p: intersect coordinates array of shape (n,3)
    :param sz: scalar radius of sphere
    :return d: distance to sphere surface, array of shape (n,) : -ve inside 

    https://iquilezles.org/www/articles/distfunctions/distfunctions.htm

    ::

        float sdSphere( vec3 p, float s )
        {
            return length(p)-s;
        }

    """
    assert len(p.shape) == 2 and p.shape[-1] == 3 and p.shape[0] > 0 
    d = np.sqrt(np.sum(p*p, axis=1)) - sz
    return d


def identity( instance_id, primitive_id ):
    """
    :param instance_id: 1-based instance id
    :param primitive_id: 1-based primitive id

    #  unsigned identity = ( instance_id << 16 ) | (primitive_id << 8) | ( buildinput_id << 0 )  ;
    """
    buildinput_id = primitive_id
    return  ( instance_id << 16 ) | (primitive_id << 8) | ( buildinput_id << 0 )

def pick_intersect_pixels( posi, pick_id ):
    sposi = np.where( posi[:,:,3].view(np.uint32) == pick_id )    

    height = posi.shape[0]
    width = posi.shape[1]

    #pick = np.zeros( (*posi.shape[:2], 1), dtype=np.float32 )  
    pick = np.zeros( (height, width, 1), dtype=np.float32 )  
    pick[sposi] = 1 
    #pick_posi = posi[sposi]   
    return pick 


def make_mask(posi):
    height = posi.shape[0]
    width = posi.shape[1]

    pxid = posi[:,:,3].view(np.uint32)      # pixel identity 

    instance_id   = ( pxid & 0xffff0000 ) >> 16    # all three _id are 1-based to distinguish from miss at zero
    primitive_id  = ( pxid & 0x0000ff00 ) >> 8 
    bindex = ( pxid & 0x000000ff ) >> 0 

    select = np.where( bindex == 45 )    

    mask = np.zeros( (height, width), dtype=np.float32 )  
    mask[select] = 1 
    return mask 


class IAS(object):
    @classmethod
    def Path(cls, iasdir):
        return "%s/grid.npy" % iasdir
    @classmethod
    def Make(cls, iasdir, idn):
        path = cls.Path(iasdir)
        return cls(path, idn) if os.path.exists(path) else None
    def __init__(self, path, idn):
        self.dir = os.path.dirname(path)
        raw = load_(path)

        # see Identity.h  all _idx are 0-based
        identity  = raw[:,0,3].view(np.uint32)  

        ins_id = idn.ins_id(identity)
        gas_id = idn.gas_id(identity)
        ins_idx = ins_id - 1 
        gas_idx = gas_id - 1 

        trs = raw.copy()
        trs[:,0,3] = 0.   # scrub the identity info 
        trs[:,1,3] = 0.
        trs[:,2,3] = 0.
        trs[:,3,3] = 1.
        itrs = np.linalg.inv(trs)  ## invert all the IAS transforms at once

        self.raw = raw
        self.trs = trs
        self.itrs = itrs
        self.ins_idx = ins_idx
        self.gas_idx = gas_idx
    pass



class GAS(object):
    def __init__(self, gasdir):
        self.dir = gasdir
        self.param = load_("%s/param.npy" % gasdir)
        self.aabb = load_("%s/aabb.npy" % gasdir)
    pass
    def __repr__(self):
        return "GAS %s param %s aabb %s " % (self.dir, str(self.param.shape), str(self.aabb.shape))

    def par(self, prim_idx):
        assert prim_idx < len(self.param)
        return self.param[prim_idx]

    def radius(self, prim_idx):
        """
        specific to sphere param layout 
        """
        p = self.par(prim_idx)
        return p[0,0]

    def sdf(self, prim_idx, lpos):
        """
        shapes need type codes and a switch statement to pick the right param and sdf functions  
        """
        radius = self.radius(prim_idx)
        d = sdf_sphere(lpos[:,:3], radius )  # sdf : distances to sphere surface 
        return d 


class Identity(object):
    def __init__(self, base):
        log.info("base:%s" % base)
        spec = load_("%s/spec.npy" % base)
        ins_bits = spec[2]
        gas_bits = spec[3]
        ins_mask = ( 1 << ins_bits ) - 1 
        gas_mask = ( 1 << gas_bits ) - 1 

        self.spec = spec
        self.ins_bits = ins_bits
        self.gas_bits = gas_bits
        self.ins_mask = ins_mask
        self.gas_mask = gas_mask

    def ins_id(self, pxid):
        ins_mask = self.ins_mask
        gas_bits = self.gas_bits  
        return (( (ins_mask << gas_bits ) & pxid ) >> gas_bits ) 
    def gas_id(self, pxid):
        gas_bits = self.gas_bits  
        gas_mask = self.gas_mask
        return  ( gas_mask & pxid ) >>  0 
    def prim_id(self, pxid):
        return ( pxid & 0xff000000 ) >> 24


class Geo(object):
    def __init__(self, base):
        log.info("base:%s" % base)
        idn = Identity(base)
        ias_dirs = sorted(glob.glob("%s/grid/*" % base))
        log.info("ias_dirs:\n%s" % "\n".join(ias_dirs)) 
        ias = {}
        for ias_idx in range(len(ias_dirs)):
            ias_ = IAS.Make(ias_dirs[ias_idx], idn)
            if ias_ is None: continue
            ias[ias_idx] = ias_
        pass

        gas_dirs = sorted(glob.glob("%s/shape/*" % base))
        log.info("gas_dirs:\n%s" % "\n".join(gas_dirs)) 
        gas = {}
        for gas_idx in range(len(gas_dirs)):
            gas[gas_idx] = GAS(gas_dirs[gas_idx])
        pass
        self.idn = idn
        self.ias = ias
        self.gas = gas

    def ins_id(self, pxid):
        return self.idn.ins_id(pxid)
    def gas_id(self, pxid):
        return self.idn.gas_id(pxid)
    def prim_id(self, pxid):
        return self.idn.prim_id(pxid)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    base = dir_()
    print(base)
    geo = Geo(base)

    i0 = geo.ias[0]

    posi = load_("posi.npy")
    hposi = posi[posi[:,:,3] != 0 ]  
    iposi = hposi[:,3].view(np.uint32)  

    #plot3d( hposi[:,:3] )
    #pick_id = identity( 500, 1 ) 
    #pick = pick_intersect_pixels(posi, pick_id )

    pxid = posi[:,:,3].view(np.uint32)      # pixel identity 

    ins_id = geo.ins_id(pxid)
    gas_id = geo.gas_id(pxid)
    prim_id = geo.prim_id(pxid)
  

    #assert np.all( layer_id == primitive_id )
    #plot2d(shape_id) 
    #plot2d(layer_id) 

    #assert np.all( primitive_id == buildinput_id ) 

    # identities of all intersected pieces of geometry 
    upxid, upxid_counts = np.unique(pxid, return_counts=True) 
    
    ires = np.zeros( (len(upxid), 4), dtype=np.int32 )
    fres = np.zeros( (len(upxid), 4), dtype=np.float32 )

if 1:
    # loop over all identified pieces of geometry with intersects
    for i in range(1,len(upxid)):   # NB zero is skipped : this assumes there are some misses 
        zid = upxid[i] 
        zid_count = upxid_counts[i]
        assert zid > 0, "must skip misses at i=0"     

        zins_id = geo.ins_id(zid)
        zgas_id = geo.gas_id(zid)
        zprim_id = geo.prim_id(zid)

        zgas_idx = zgas_id - 1 
        zinstance_idx = zins_id - 1  
        zprimitive_idx = zprim_id - 1 

        tr = geo.ias[0].trs[zinstance_idx]
        itr = geo.ias[0].itrs[zinstance_idx]

        gas_idx = geo.ias[0].gas_idx[zinstance_idx]   # lookup in the IAS the gas_idx for this instance
        ins_idx = geo.ias[0].ins_idx[zinstance_idx]   # lookup in the IAS the ins_idx for this instance  

        assert ins_idx == zinstance_idx  
        assert gas_idx == zgas_idx 

        z = np.where(pxid == zid)   

        zpxid = posi[z][:,3].view(np.uint32).copy()
        zposi = posi[z].copy()  
        zposi[:,3] = 1.      # global 3d coords for intersect pixels, ready for transform
        zlpos = np.dot( zposi, itr ) # transform global positions into instance local ones 

        radius = geo.gas[gas_idx].radius(zprimitive_idx)  
        d = geo.gas[gas_idx].sdf(zprimitive_idx, zlpos[:,:3] )  

         

        print("i:%5d zid:%9d zid_count:%6d ins_idx:%4d gas_idx:%3d prim_idx:%3d  d.min:%10s d.max:%10s radius:%s  "  % ( i, zid, zid_count, ins_idx, gas_idx, zprimitive_idx, d.min(), d.max(), radius ))
        pass
        fres[i] = (d.min(),d.max(), radius,0. )
        ires[i] = ( len(zposi), ins_idx, gas_idx, zprimitive_idx )
    pass
   
    print("ires\n", ires) 
    print("fres\n", fres) 
    abs_dmax = np.max(np.abs(fres[:,:2]))   
    print("abs_dmax:%s" % abs_dmax)
    print(dir_())

if 0:
    pick_plot = False
    fig, axs = plt.subplots(3 if pick_plot else 2 )
    #axs[0].imshow(instance_id, vmin=0, vmax=10)  # big sphere is there, just not visible as too much range
    axs[0].imshow(instance_id) 
    axs[0].set_xlabel("instance_id %s %s " % (instance_id.min(),instance_id.max()))     
    axs[1].imshow(primitive_id) 
    axs[1].set_xlabel("primitive_id %s %s " % (primitive_id.min(),primitive_id.max()))     

    if pick_plot:
        axs[2].imshow(pick) 
    pass
    fig.show()


