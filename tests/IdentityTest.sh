#!/bin/bash -l

name=IdentityTest 

gcc $name.cc \
   -I.. \
   -lstdc++ -o /tmp/$name && /tmp/$name



