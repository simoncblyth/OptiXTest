// ./FoundryTest.sh

#include <iostream>
#include "sutil_vec_math.h"

#include "Foundry.h"

int main(int argc, char** argv)
{
    Foundry fd ;  
    //fd.init(); 
    //fd.makeSphere(); 
    unsigned s0 = fd.makeLayered("sphere", 100.f, 10 ); 
    unsigned s1 = fd.makeLayered("sphere", 1000.f, 10 ); 
    unsigned s2 = fd.makeLayered("sphere", 50.f, 5 ); 

    fd.dump(); 

    std::vector<float> aabb ; 
    fd.get_aabb(aabb, s0 ); 
    Foundry::Dump(aabb, "s0"); 

    fd.get_aabb(aabb, s2 ); 
    Foundry::Dump(aabb, "s2"); 

    fd.get_aabb(aabb, s1 ); 
    Foundry::Dump(aabb, "s1"); 

    fd.write("/tmp", "FoundryTest_" ); 


    return 0 ; 
}
