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
    cd OptiXTest
    ./go.sh # gets glm, builds, runs -> ppm image file    
     

Links
--------

* https://simoncblyth.bitbucket.io
* https://bitbucket.org/simoncblyth/opticks



