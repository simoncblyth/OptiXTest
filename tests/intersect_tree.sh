#!/bin/bash -l 

name=intersect_tree 
srcs="$name.cc ../Node.cc"

gcc $srcs \
    -I.. \
    -DDEBUG=1 \
    -std=c++11 \
    -lstdc++ \
    -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name



