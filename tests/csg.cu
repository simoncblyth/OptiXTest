//  name=csg ; nvcc $name.cu -ccbin=/usr/bin/clang -DDEBUG=1 -std=c++11 -lstdc++ -I/usr/local/cuda/include  -o /tmp/$name && /tmp/$name

#ifdef DEBUG
#include "stdio.h"
#endif

#include "sutil_vec_math.h"
#include "error.h"
#include "csg.h"

int main(int argc, char** argv)
{
    float4 isect = make_float4( 1.f, 2.f, 3.f, 4.f ); ;

    CSG_Stack csg ;  
    csg.curr = -1 ; 

    int ierr = 0 ; 
    unsigned long long bef, aft ; 

    for(int i=0 ; i < 20 ; i++)
    {
       unsigned nodeIdx = i  ; 
       bef = csg_repr(csg) ; 
       ierr = csg_push( csg, isect,  nodeIdx );
       aft = csg_repr(csg) ; 
#ifdef DEBUG
       printf( "i %2d bef %16llx  aft %16llx \n", i, bef, aft ); 
#endif
    }
    return ierr ; 
}

