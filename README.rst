OptiXTest
==============

.. contents:: Table Of Contents


Overview
----------

Cutting edge has moved to the below two repos with the CSG model 
and rendering of that with OptiX moved into separate packages.

* https://github.com/simoncblyth/CSGOptiX
* https://github.com/simoncblyth/CSG

History
----------

This repo was used as a playground for learning/investigating OptiX 7 techniques needed by Opticks.
Hopefully also some of the OptiX 7 interface classes will subsequenly become part of Opticks. 
This started from the Opticks example:

* https://bitbucket.org/simoncblyth/opticks/src/master/examples/UseOptiX7GeometryInstancedGASCompDyn/


Standalone Building
---------------------

Depends only on NVIDIA OptiX 7.0::

    export OPTIX_PREFIX=/usr/local/OptiX_700    # (might need to put this in .bashrc/.bash_profile)

    git clone https://github.com/simoncblyth/OptiXTest 
    git clone git@github.com:simoncblyth/OptiXTest.git     # ssh clone for easy key based committing 

    cd OptiXTest
    ./go.sh # gets glm, builds, runs -> ppm image file    
     

Open Questions
-----------------

How to handle remainder non-repeated geometry ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* suspect clumping into one remainder GAS non-optimal  
* could have special handling of GMergedMesh zero creating many separate GAS
* probably some kind of futher instancing algorithm with looser criteria could 
  be applied to the remainder volumes 
* could change the original instancing to find more instances within the "remainder"

TODO: experiment with changing the instancing criteria to catch more and reduce the remainder

* for JUNO only a handful of non-repeated volumes are optically important, 
  they probably deserve special treatment 


How to handle rendering and simulation of exactly the same geometry ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* HitGroup and Miss shaders and SBT records exactly the same. Just needs different RayGen ? 

How to switch ?

* GPU runtime parameter switch ?  NO: should always aim to simplify and reduce what is on GPU 
* different ptx ? YES: as minimizing the amount of ptx(even unused) is good   
* split the ptx for geometry and raygen and have separate rg_camera and rg_simulation ptx 
* a CPU side parameter then controls which flavor of pipeline is constructed and 
  how that is managed : ie what data needs uploading/downloading 

  * rendering : upload viewpoint, download pixels
  * simulation : upload gensteps, download hits 



Classes
---------


Geometry Model Classes
~~~~~~~~~~~~~~~~~~~~~~~~

Geo.h
   container for shapes and grids

Grid.h
   configurably prepares vector of transforms with shape references 

Shape.h
   holder of aabb and param for multi-prim(aka multi-layer) shapes
   perhaps Comp would be a better name (short for Composite but that 
   is too similar to Composition)

InstanceId.h
   identity bit packer

Identity.h
   another identity bit packer


Bring in Opticks Geometry struct 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

quad.h
    union trick

Prim.h
    q0 wrapper for access to offsets  



OptiX 7 Geometry 
~~~~~~~~~~~~~~~~~~~~~

AS.h
    base struct for GAS and IAS

BI.h
    holds OptixBuildInput and workings 

GAS.h
    holds reference to the source Shape and vector of BI

GAS_Builder.h
    converts the Shape into BI and thence GAS

IAS.h
    vector of glm::mat4 and d_instances 

IAS_Builder.h
    converts Grid with gas_idx instrumented transforms into IAS

PIP.h
    OptixProgramGroup and OptixPipeline

SBT.h
    nexus of geometry control holding OptixShaderBindingTable 

OptiX 7 Others
~~~~~~~~~~~~~~~~~

Ctx.h
    holder of OptixDeviceContext and Params with uploadParams

Properties.h
    optix limits obtained with optixDeviceContextGetProperty 

OPTIX_CHECK.h
    preprocessor call wrapper and exception 

Frame.h
    holder of pixels and isect data

Binding.h
    host/device types

Params.h
    host/device view params 

OptiX 6
~~~~~~~~~~

Six.h
    one struct renderer


CUDA Misc
~~~~~~~~~~~~

CUDA_CHECK.h
    preprocessor call wrapper and exception 

sutil_vec_math.h
    lerp roundUp etc..
 
sutil_Preprocessor.h
    needed by sutil_vec_math.h

Image Handling 
~~~~~~~~~~~~~~~

SPPM.h
   ppm writing 

SIMG.hh
   jpg png writing using stb_image.h stb_image_write.h

Utilities
~~~~~~~~~~~

Sys.h
   unsigned_as_float float_as_unsigned 

Util.h
   misc  

NP.hh
   array persistency in NPY format, NumPy readable  

NPU.hh
   required by NP.hh



Splitting off CSG Model into separate pkg/repo for reusability
-----------------------------------------------------------------

Point of the exercise is to end up with as little as possible 
in the package that depends on OptiX 7 making the geomerty 
model maximally reusable. Including the ability to test the
intersection on CPU.


::

    ## basis types used by geometry model 

    sutil_vec_math.h
    qat4.h
    Quad.h

    ## utilities used by geometry model

    CU.h
    CUDA_CHECK.h
    Sys.h

    ## intersection underpinnings

    robust_quadratic_roots.h
    intersect_node.h

    error.h
    tranche.h
    csg.h
    pack.h
    csg_classify.h
    postorder.h

    intersect_tree.h

    ## main players of geometry model 

    Node.h
    Prim.h
    PrimSpec.h
    OpticksCSG.h
    Solid.h

    Tran.h
    AABB.h
    Util.h
    Foundry.h

    ## higher level geometry 

    Geo.h
    InstanceId.h
    Grid.h

    ## test machinery 

    Scan.h


    ## unused/superceeded, delete ?

    history.h
    Shape.h       # delete Shape.h after Six.cc is updated to new model
    Identity.h

    ## utilities npy/jpg/...

    NP.hh
    NPU.hh
    SIMG.hh
    stb_image.h
    stb_image_write.h

    ## image mechanics
    Frame.h

    ## 3D math
    View.h

    ## OptiX 7

    Ctx.h
    Properties.h
    Params.h
    Binding.h
    OPTIX_CHECK.h
    GAS.h
    GAS_Builder.h
    IAS.h
    IAS_Builder.h
    PIP.h
    SBT.h
    AS.h
    BI.h

    ## OptiX 6

    Six.h





WIP : rethink identity handling re:gas_idx 
-----------------------------------------------------------

It is inconvenient to have to lookup the gas_idx in the IAS. Where to encode gas_idx ?
Better to not require an attribute/register for this if possible.

* optixGetInstanceId() limited to 3 bytes: 0xffffff (24 bits)
  currently are splitting that 14 bits for instance_id and 10 bits for gas_id 

* optixGetPrimitiveIndex() is also controllable with a bias primitiveIndexOffset in GAS_Builder::MakeCustomPrimitivesBI_11N


::

    In [9]: 0xfff
    Out[9]: 4095

    In [10]: 0xffffffff
    Out[10]: 4294967295

    In [11]: 0xfffff
    Out[11]: 1048575



Links
--------

* https://simoncblyth.bitbucket.io
* https://bitbucket.org/simoncblyth/opticks



