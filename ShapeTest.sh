#!/bin/bash -l 
source ./env.sh 
name=ShapeTest 
mkdir -p /tmp/ShapeTestWrite/{0,1} 
gcc -g $name.cc Node.cc Shape.cc Sys.cc \
      -lstdc++ -std=c++11 \
        -I. \
        -I$PREFIX/externals/glm/glm \
        -I/usr/local/cuda/include \
        -o /tmp/$name &&  lldb_ /tmp/$name

