#!/usr/bin/env python


import os, sys, glob, logging, codecs
d_ = lambda _:codecs.latin_1_decode(_)[0]  
log = logging.getLogger(__name__)

from OpticksCSG import CSG_ as CSG 

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
    def Make(cls, iasdir, spec):
        path = cls.Path(iasdir)
        return cls(path, spec) if os.path.exists(path) else None
    def __init__(self, path, spec):
        self.dir = os.path.dirname(path)
        raw = load_(path)

        # see Identity.h  all _idx are 0-based
        identity  = raw[:,0,3].view(np.uint32)  

        ins_id = spec.ins_id(identity)
        gas_id = spec.gas_id(identity)
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


class Foundry(object):
    qwn = "solid node prim plan itra tran".split()
    def __init__(self, dir_):
        self.dir = dir_
        for qwn in self.qwn:
            a = load_("%s/%s.npy" % (dir_,qwn))
            setattr(self, qwn, a )
        pass    
    pass

class GAS(object):
    def __init__(self, gasdir):
        """
        
        """
        self.dir = gasdir
        self.node = load_("%s/node.npy" % gasdir)
        self.prim = load_("%s/prim.npy" % gasdir)
        self.aabb = load_("%s/aabb.npy" % gasdir)
    pass
    def __repr__(self):
        return "GAS %s node %s aabb %s prim %s " % (self.dir, str(self.node.shape), str(self.aabb.shape), str(self.prim.shape))

    def radius(self, prim_idx):
        """specific to sphere Node layout"""
        nd = self.node[prim_idx]
        return nd[0,3]

    def typecode(self, prim_idx):
        """
        hmm for now assuming one node per prim, when generalize to 
        having node trees will need to have a node_idx argument 
        and consult the prim array to get the node offsets to use
        in order to pluck from the tree of nodes for each gas 
        """
        tc = self.node[prim_idx].view(np.uint32)[2,3]
        return tc 

    def sdf(self, prim_idx, lpos):
        tc = self.typecode(prim_idx)
        if tc == CSG.SPHERE:
            radius = self.radius(prim_idx)
            d = sdf_sphere(lpos[:,:3], radius )  # sdf : distances to sphere surface 
        else:
            assert 0 
        pass
        return d 


class Spec(object):
    """
    From OptiX7Test.cu:__closesthit__ch::

       unsigned instance_id = optixGetInstanceId() ; 
       unsigned prim_id  = 1u + optixGetPrimitiveIndex() ;
       unsigned identity = (( prim_id & 0xff ) << 24 ) | ( instance_id & 0x00ffffff ) ;

    When the intersect is not on an instance the value returned in ~0u (ie -1) 
    which "fills" whatever mask. 
    """
    def __init__(self, base):
        log.info("base:%s" % base)
        uspec = load_("%s/uspec.npy" % base)
        fspec = load_("%s/fspec.npy" % base)

        num_solid = uspec[0]
        num_grid = uspec[1]
        ins_bits = uspec[2]
        gas_bits = uspec[3]
        ins_mask = ( 1 << ins_bits ) - 1 
        gas_mask = ( 1 << gas_bits ) - 1 
        top = str(d_(uspec[4]))[:2]   ## eg i0 i1 g0  

        self.uspec = uspec
        self.fspec = fspec
        self.ce = fspec 
        self.num_solid = num_solid
        self.num_grid = num_grid
        self.ins_bits = ins_bits
        self.gas_bits = gas_bits
        self.ins_mask = ins_mask
        self.gas_mask = gas_mask
        self.top = top 

    def ins_id(self, pxid):
        """

        """
        ins_mask = self.ins_mask
        gas_bits = self.gas_bits  
        return (( (ins_mask << gas_bits ) & pxid ) >> gas_bits ) 
    def gas_id(self, pxid):
        gas_bits = self.gas_bits  
        gas_mask = self.gas_mask
        return  ( gas_mask & pxid ) >>  0 
    def prim_id(self, pxid):
        return ( pxid & 0xff000000 ) >> 24

    def __repr__(self):
        fmt = "Spec num_solid %d num_grid %d ins_bits %d ins_mask %x gas_bits %d gas_mask %x top %s ce %r " 
        return fmt % (self.num_solid, self.num_grid, self.ins_bits, self.ins_mask, self.gas_bits, self.gas_mask, self.top, self.ce )



class Geo(object):
    def __init__(self, base):
        log.info("base:%s" % base)
        spec = Spec(base)
        log.info("spec:%r" % spec)
        ias_dirs = sorted(glob.glob("%s/grid/*" % base))
        log.info("ias_dirs:\n%s" % "\n".join(ias_dirs)) 
        ias = {}
        for ias_idx in range(len(ias_dirs)):
            ias_ = IAS.Make(ias_dirs[ias_idx], spec)
            if ias_ is None: continue
            ias[ias_idx] = ias_
        pass
        self.spec = spec
        self.ias = ias
        self.fdr = Foundry(os.path.join(base, "foundry"))

    def load_gas(self, base):
        """no longer using ?"""
        gas_dirs = sorted(glob.glob("%s/shape/*" % base))
        log.info("gas_dirs:\n%s" % "\n".join(gas_dirs)) 
        gas = {}
        for gas_idx in range(len(gas_dirs)):
            gas[gas_idx] = GAS(gas_dirs[gas_idx])
        pass
        return gas

    def ins_id(self, pxid):
        return self.spec.ins_id(pxid)
    def gas_id(self, pxid):
        return self.spec.gas_id(pxid)
    def prim_id(self, pxid):
        return self.spec.prim_id(pxid)


class IntersectCheck(object):
    def __init__(self, geo):
        self.geo = geo
    def check(self, posi):
        geo = self.geo
        pxid = posi[:,:,3].view(np.uint32) # pixel identity 

        ins_id = geo.ins_id(pxid)
        gas_id = geo.gas_id(pxid)
        prim_id = geo.prim_id(pxid)

        # identities of all intersected pieces of geometry 
        upxid, upxid_counts = np.unique(pxid, return_counts=True) 
        
        ires = np.zeros( (len(upxid), 4), dtype=np.int32 )
        fres = np.zeros( (len(upxid), 4), dtype=np.float32 )

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
     
            fmt = "i:%5d zid:%9d zid_count:%6d ins_idx:%4d gas_idx:%3d prim_idx:%3d  d.min:%10s d.max:%10s radius:%s  " 
            print(fmt  % ( i, zid, zid_count, ins_idx, gas_idx, zprimitive_idx, d.min(), d.max(), radius ))
            pass
            fres[i] = (d.min(),d.max(), radius,0. )
            ires[i] = ( len(zposi), ins_idx, gas_idx, zprimitive_idx )
        pass
        print("ires\n", ires) 
        print("fres\n", fres) 
        abs_dmax = np.max(np.abs(fres[:,:2]))   
        print("abs_dmax:%s" % abs_dmax)
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if not 'OUTDIR' in os.environ:
        print("use ./posi.sh to setup environment")
        sys.exit(0)
    else:
        print("OUTDIR:%s" % os.environ['OUTDIR']) 
    pass
    base = dir_()
    print(base)

    geo = Geo(base)
    ick = IntersectCheck(geo)

    posi = load_("posi.npy")
    pxid = posi[:,:,3].view(np.uint32) # pixel identity, 0 for miss 
    hposi = posi[pxid > 0]  

    for k in range(3):
        print("hposi[:,%d].min()/.max() %10.4f %10.4f " % (k,hposi[:,k].min(),hposi[:,k].max())) 
    pass

    iposi = hposi[:,3].view(np.uint32)  

    num_pixels = pxid.size
    num_hits = np.count_nonzero(pxid)
    hit_fraction = float(num_hits)/float(num_pixels)   
    print("num_pixels:%d num_hits:%d hit_fraction:%8.4f " % (num_pixels, num_hits, hit_fraction ))

    #plot3d( hposi[:,:3] )
    #pick_id = identity( 500, 1 ) 
    #pick = pick_intersect_pixels(posi, pick_id )

    #assert np.all( layer_id == primitive_id )
    #plot2d(shape_id) 
    #plot2d(layer_id) 

    #assert np.all( primitive_id == buildinput_id ) 

    #ick.check(posi)
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
pass
