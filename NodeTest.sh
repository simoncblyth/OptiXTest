#!/bin/bash -l 
source ./env.sh 
name=NodeTest ; gcc -g $name.cc Node.cc -lstdc++ -std=c++11 -I. -I$PREFIX/externals/glm/glm -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

