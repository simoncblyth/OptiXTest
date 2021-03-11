#!/bin/bash -l 

spec=$1

source ./env.sh 

echo RM OUTDIR $OUTDIR
rm -rf $OUTDIR
mkdir -p $OUTDIR

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

ppm=$OUTDIR/pixels.ppm
npy=$OUTDIR/posi.npy

echo BIN    : $BIN 
echo OUTDIR : $OUTDIR
echo spec : $spec
echo ppm  : $ppm

if [ "$(uname)" == "Linux" ]; then 
   dig=$(cat $ppm | md5sum)
else
   dig=$(cat $ppm | md5)
fi 
echo md5  : $dig
echo npy  : $npy
ls -l $ppm $npy 

open $ppm

exit 0

