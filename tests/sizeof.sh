#!/bin/bash -l 

name=sizeof 
gcc $name.cc -I.. -std=c++11 -I/usr/local/cuda/include -lstdc++ -o /tmp/$name && /tmp/$name

