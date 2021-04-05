#!/bin/bash -l

name=csg 
gcc $name.cc \
     -I.. \
     -std=c++11 \
     -lstdc++ \
     -I/usr/local/cuda/include  -o /tmp/$name && /tmp/$name



