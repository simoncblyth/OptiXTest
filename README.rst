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
     

TODO : rethink identity handling re:gas_idx 
-----------------------------------------------------------

::

    520 ///
    521 /// In Intersection and AH this corresponds to the currently intersected primitive.
    522 /// In CH this corresponds to the primitive index of the closest intersected primitive.
    523 /// In EX with exception code OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_HIT_SBT corresponds to the active primitive index. Returns zero for all other exceptions.
    524 static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex();
    525 
    526 
    527 /// Returns the OptixInstance::instanceId of the instance within the top level acceleration structure associated with the current intersection.
    528 ///
    529 /// When building an acceleration structure using OptixBuildInputInstanceArray each OptixInstance has a user supplied instanceId.
    530 /// OptixInstance objects reference another acceleration structure.  During traversal the acceleration structures are visited top down.
    531 /// In the Intersection and AH programs the OptixInstance::instanceId corresponding to the most recently visited OptixInstance is returned when calling optixGetInstanceId().
    532 /// In CH optixGetInstanceId() returns the OptixInstance::instanceId when the hit was recorded with optixReportIntersection.
    533 /// In the case where there is no OptixInstance visited, optixGetInstanceId returns ~0u
    534 static __forceinline__ __device__ unsigned int optixGetInstanceId();
    535 
    536 /// Returns the zero-based index of the instance within its instance acceleration structure associated with the current intersection.
    537 ///
    538 /// In the Intersection and AH programs the index corresponding to the most recently visited OptixInstance is returned when calling optixGetInstanceIndex().
    539 /// In CH optixGetInstanceIndex() returns the index when the hit was recorded with optixReportIntersection.
    540 /// In the case where there is no OptixInstance visited, optixGetInstanceId returns 0
    541 static __forceinline__ __device__ unsigned int optixGetInstanceIndex();
    542 


It is inconvenient to have to lookup the gas_idx in the IAS. Where to encode gas_idx ?

Options:

1. hijack optixGetPrimitiveIndex() setting primitiveIndexOffset in GAS_Builder::MakeCustomPrimitivesBI_11N 
2. hijack optixGetInstanceId()

   * full unsigned int for instance_id can be 


::

    In [9]: 0xfff
    Out[9]: 4095

    In [10]: 0xffffffff
    Out[10]: 4294967295

    In [11]: 0xfffff
    Out[11]: 1048575




Thus can then remove the limited bindex encoding that will not scale to relistic 
numbers of GAS.


Links
--------

* https://simoncblyth.bitbucket.io
* https://bitbucket.org/simoncblyth/opticks



