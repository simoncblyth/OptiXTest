// ./ScanTest.sh

#include <vector>
#include <cassert>
#include <iostream>

#include "sutil_vec_math.h"
#include "Foundry.h"
#include "Solid.h"
#include "Scan.h"

int main(int argc, char** argv)
{
    const char* dir = "/tmp/ScanTest_scans" ; 
    Foundry fd ;  

    for(unsigned i=0 ; i < fd.getNumSolid() ; i++)
    {
        const Solid* solid = fd.getSolid(i); 

        Scan sc(dir, &fd, solid); 
        sc.axis_scan(); 
        sc.rectangle_scan(); 
        sc.circle_scan(); 
    }
    return 0 ;  
}
