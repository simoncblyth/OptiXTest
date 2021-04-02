#!/bin/bash 

sdir=$(pwd)
name=$(basename $sdir)



export PREFIX=/tmp/$USER/opticks/$name
source $PREFIX/build/buildenv.sh 

export PATH=$PREFIX/bin:$PATH
export BIN=$(which $name)

#tmin=2.0
#tmin=1.5
#tmin=1.0
#tmin=0.5
tmin=0.1

geometry=parade
#geometry=sphere_containing_grid_of_spheres
#geometry=layered_sphere
#geometry=layered_zsphere

#geometry=sphere
#geometry=zsphere
#geometry=cone
#geometry=hyperboloid
#geometry=box3
#geometry=plane
#geometry=slab
#geometry=cylinder
#geometry=disc
#geometry=convexpolyhedron_cube
#geometry=convexpolyhedron_tetrahedron


gridmodulo=0,1,2,3,4,5,6,7,8,9,10
#gridsingle=2
gridsingle=""

#gridspec=-10:11:2,-10:11:2,-10:11:2
gridspec=-10:11:2,-10:11:2,0:8:2
#gridspec=-40:41:4,-40:41:4,-40:41:4
#gridspec=-40:41:10,-40:41:10,-40:41:10
#gridspec=-40:41:10,-40:41:10,0:1:1

gridscale=200.0



#eye=-0.5,-0.5,0.0
eye=-0.5,0.0,0.15
#eye=-0.5,-0.5,-0.5
#eye=-1.0,-1.0,0.0
#eye=-1.0,-1.0,0.5

cameratype=0
#cameratype=1

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
export GRIDMODULO=${GRIDMODULO:-$gridmodulo}
export GRIDSINGLE=${GRIDSINGLE:-$gridsingle}
export GRIDSCALE=${GRIDSCALE:-$gridscale}

export EYE=${EYE:-$eye} 
export LAYERS=${LAYERS:-$layers}
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
printf "$fmt" GRIDMODULO $GRIDMODULO
printf "$fmt" GRIDSINGLE $GRIDSINGLE
printf "$fmt" GRIDSCALE $GRIDSCALE

printf "$fmt" EYE $EYE
printf "$fmt" LAYERS $LAYERS
printf "$fmt" OUTDIR $OUTDIR

