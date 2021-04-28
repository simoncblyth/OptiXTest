#!/bin/bash -l 

spec=$1

source ./env.sh 

echo RM OUTDIR $OUTDIR
rm -rf $OUTDIR
mkdir -p $OUTDIR

mkdir -p $OUTDIR/shape/0
mkdir -p $OUTDIR/shape/1
mkdir -p $OUTDIR/shape/2

mkdir -p $OUTDIR/grid/0
mkdir -p $OUTDIR/grid/1
mkdir -p $OUTDIR/grid/2

mkdir -p $OUTDIR/foundry


echo $0 

if [ -n "$DEBUG" ]; then 
    if [ "$(uname)" == "Linux" ]; then
       gdb -ex r --args $BIN $spec
    elif [ "$(uname)" == "Darwin" ]; then
       lldb_ $BIN $spec
    fi
else
    $BIN $spec
fi 

[ $? -ne 0 ] && echo $0 : run  FAIL && exit 3

jpg=$OUTDIR/pixels.jpg
npy=$OUTDIR/posi.npy

echo BIN    : $BIN 
echo OUTDIR : $OUTDIR
echo spec : $spec
echo jpg  : $jpg

if [ "$(uname)" == "Linux" ]; then 
   dig=$(cat $jpg | md5sum)
else
   dig=$(cat $jpg | md5)
fi 
echo md5  : $dig
echo npy  : $npy
ls -l $jpg $npy 

if [ "$(uname)" == "Darwin" ]; then
    open $jpg
fi
exit 0

