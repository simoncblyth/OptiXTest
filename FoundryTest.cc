// ./FoundryTest.sh

#include <iostream>
#include <cassert>

#include "sutil_vec_math.h"
#include "Foundry.h"


void test_layered()
{
    Foundry fd ;  

    Solid* s0 = fd.makeLayered("sphere", 100.f, 10 ); 
    Solid* s1 = fd.makeLayered("sphere", 1000.f, 10 ); 
    Solid* s2 = fd.makeLayered("sphere", 50.f, 5 ); 
    Solid* s3 = fd.makeSphere() ; 

    fd.dump(); 

    assert( fd.getSolidIdx(s0) == 0 ); 
    assert( fd.getSolidIdx(s1) == 1 ); 
    assert( fd.getSolidIdx(s2) == 2 ); 
    assert( fd.getSolidIdx(s3) == 3 ); 

    fd.write("/tmp", "FoundryTest_" ); 
}



int main(int argc, char** argv)
{
    //test_layered(); 

    Foundry fd ; 
    fd.makeDemoSolids(); 
    for(unsigned i = 0 ; i < fd.solid.size() ; i++ )
    {
        unsigned solidIdx = i ; 
        std::cout << "solidIdx " << solidIdx << std::endl ; 
        PrimSpec ps = fd.getPrimSpec(solidIdx);
        ps.dump(""); 
    }

    std::string bmap = fd.getBashMap(); 
    std::cout  << bmap << std::endl ; 

    return 0 ; 
}
