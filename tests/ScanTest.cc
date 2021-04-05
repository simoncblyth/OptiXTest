// ./ScanTest.sh

#include <vector>
#include <cassert>
#include <iostream>

#include "sutil_vec_math.h"
#include "Foundry.h"
#include "Solid.h"
#include "Scan.h"
#include "Geo.h"


void test_Foundry_Scan()
{
    const char* dir = "/tmp/ScanTest_scans" ; 
    Foundry fd ;  

    //fd.makeDemoSolids(); 
    fd.makeEllipsoid(); 
    //const char* name = "sphe" ; 
    //fd.makeClustered(name, 0,1,1, 0,1,1,  0,2,1, 100. );  

    unsigned numSolid = fd.getNumSolid() ; 
    std::cout << "numSolid " << numSolid << std::endl ; 

    for(unsigned i=0 ; i < numSolid ; i++)
    {
        const Solid* solid = fd.getSolid(i); 

        Scan sc(dir, &fd, solid); 
        sc.axis_scan(); 
        sc.rectangle_scan(); 
        sc.circle_scan(); 
    }
}


int main(int argc, char** argv)
{
    Foundry foundry ;  
    Geo geo(&foundry) ; 
    float top_extent = geo.getTopExtent() ;
    std::cout << "top_extent " << top_extent << std::endl ; 

    const char* dir = "/tmp/ScanTest_scans" ; 

    const Solid* solid0 = foundry.getSolid(0); 

    Scan scan(dir, &foundry, solid0 ); 
    scan.axis_scan() ; 
    scan.rectangle_scan() ; 
    scan.circle_scan() ; 

    return 0 ;  
}
