// ./FoundryTest.sh

#include <iostream>
#include "sutil_vec_math.h"

#include "Foundry.h"

int main(int argc, char** argv)
{
    Foundry fd ;  
    fd.init(); 

    const Solid* so = fd.getSolid("slab"); 
    std::cout << so->desc() << std::endl ; 

    for(unsigned primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const Prim* pr = fd.getPrim(primIdx);   // numNode,nodeOffset,tranOffset,planOffset
        std::cout << pr->desc() << std::endl ; 

        for(unsigned nodeIdx=pr->nodeOffset ; nodeIdx < pr->nodeOffset+pr->numNode ; nodeIdx++)
        {
            const Node* nd = fd.getNode(nodeIdx); 
            std::cout << nd->desc() << std::endl ; 
        }
    } 


    return 0 ; 
}
