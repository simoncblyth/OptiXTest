// ./FoundryTest.sh

#include <iostream>
#include <cassert>

#include "sutil_vec_math.h"
#include "Foundry.h"

int main(int argc, char** argv)
{
    Foundry fd ;  
    fd.solid.reserve(100);  

    //fd.init(); 
    Solid* s0 = fd.makeLayered("sphere", 100.f, 10 ); 
    Solid* s1 = fd.makeLayered("sphere", 1000.f, 10 ); 
    Solid* s2 = fd.makeLayered("sphere", 50.f, 5 ); 
    Solid* s3 = fd.makeSphere() ; 

    fd.dump(); 

    assert( fd.getSolidIdx(s0) == 0 ); 
    assert( fd.getSolidIdx(s1) == 1 ); 
    assert( fd.getSolidIdx(s2) == 2 ); 
    assert( fd.getSolidIdx(s3) == 3 ); 

    unsigned solidIdx = 3 ;     

    unsigned num_prim = 0 ; 
    unsigned stride_in_bytes = 0 ; 
    const float* aabb = fd.getPrimAABB(num_prim, stride_in_bytes, solidIdx ); 

    assert( aabb && num_prim > 0 && stride_in_bytes > 0  );
 
    Foundry::Dump( aabb, num_prim, stride_in_bytes, "solid_0" ); 


    fd.write("/tmp", "FoundryTest_" ); 


    return 0 ; 
}
