// ./intersect_node.sh

#include <vector>
#include <cassert>
#include <iostream>

#include "sutil_vec_math.h"
#include "Solid.h"
#include "Scan.h"

int main(int argc, char** argv)
{
    std::vector<Solid*> solids ; 
    if(argc > 1)
    {
         
        for(int s=1 ; s < argc ; s++)
        {
            char* arg = argv[s] ; 
            std::cout << " arg " << arg << std::endl ; 
            Solid* solid = Solid::Make(arg); 
            assert( solid ); 
            solids.push_back(solid); 
        }
    }
    else
    {
        Solid::MakeSolids(solids); 
    }

    unsigned ni = solids.size() ;
    for(unsigned i=0 ; i < ni ; i++)
    {
        Solid* solid = solids[i] ; 

        Scan sc(solid); 

        sc.axis_scan(); 
        sc.rectangle_scan(); 
        sc.circle_scan(); 

    }

    return 0 ;  
}
