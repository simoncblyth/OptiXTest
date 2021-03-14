OptiXTest
==============

.. contents:: Table Of Contents


Overview
----------

This repo is used as a playground for learning/investigating OptiX 7 techniques needed by Opticks.
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



