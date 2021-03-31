#!/bin/bash 

sdir=$(pwd)
name=$(basename $sdir)



export PREFIX=/tmp/$USER/opticks/$name
source $PREFIX/build/buildenv.sh 

export PATH=$PREFIX/bin:$PATH
export BIN=$(which $name)

#tmin=2.0
tmin=1.0
#tmin=0.5
#tmin=0.1

geometry=sphere
#geometry=zsphere
#geometry=sphere_containing_grid_of_spheres

modulo=0,1
single=2
#single=""

gridspec=-10:11:2,-10:11:2,-10:11:2
#gridspec=-40:41:4,-40:41:4,-40:41:4
#gridspec=-40:41:10,-40:41:10,-40:41:10
#gridspec=-40:41:10,-40:41:10,0:1:1

#eye=-0.5,-0.5,0.0
eye=-0.5,-0.5,0.5
#eye=-0.5,-0.5,-0.5
#eye=-1.0,-1.0,1.0

# when non-zero repeats outer aabb for all layers of compound shape (optix 7 only)
kludge_outer_aabb=0
#kludge_outer_aabb=1

#gas_bi_aabb=0  # 1NN : has bbox clipping issue for multi-layer GAS  
gas_bi_aabb=1  # 11N  

cameratype=0

# number of concentric layers in compound shapes
#layers=1     
#layers=2
#layers=3
layers=20

# make sensitive to calling environment
export GEOMETRY=${GEOMETRY:-$geometry}
export TMIN=${TMIN:-$tmin}
export CAMERATYPE=${CAMERATYPE:-$cameratype}
export GRIDSPEC=${GRIDSPEC:-$gridspec}
export EYE=${EYE:-$eye} 
export MODULO=${MODULO:-$modulo}
export SINGLE=${SINGLE:-$single}
export LAYERS=${LAYERS:-$layers}
export KLUDGE_OUTER_AABB=${KLUDGE_OUTER_AABB:-$kludge_outer_aabb}
export GAS_BI_AABB=${GAS_BI_AABB:-$gas_bi_aabb}
export OUTDIR=$PREFIX/$GEOMETRY/TMIN_${TMIN}

fmt="%-20s : %s \n"
printf "$fmt" name $name
printf "$fmt" PREFIX $PREFIX
printf "$fmt" OPTIX_VERSION $OPTIX_VERSION
printf "$fmt" BIN $BIN
printf "$fmt" GEOMETRY $GEOMETRY
printf "$fmt" TMIN $TMIN
printf "$fmt" CAMERATYPE $CAMERATYPE
printf "$fmt" GRIDSPEC $GRIDSPEC
printf "$fmt" EYE $EYE
printf "$fmt" MODULO $MODULO
printf "$fmt" SINGLE $SINGLE
printf "$fmt" LAYERS $LAYERS
printf "$fmt" KLUDGE_OUTER_AABB $KLUDGE_OUTER_AABB
printf "$fmt" GAS_BI_AABB $GAS_BI_AABB
printf "$fmt" OUTDIR $OUTDIR

