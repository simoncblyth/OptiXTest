// ./ScanTest.sh

#include <vector>
#include <cassert>
#include <iostream>

#include "sutil_vec_math.h"
#include "Foundry.h"
#include "Solid.h"
#include "Scan.h"
#include "Geo.h"
#include "Util.h"
#include "View.h"


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

    unsigned width = 1280u ; 
    unsigned height = 720u ; 

    glm::vec4 eye_model ; 
    Util::GetEVec(eye_model, "EYE", "-1.0,-1.0,1.0,1.0"); 

    const float4 gce = geo.getCenterExtent() ;   
    glm::vec4 ce(gce.x,gce.y,gce.z, gce.w*1.4f );   // defines the center-extent of the region to view
    glm::vec3 eye,U,V,W  ;
    Util::GetEyeUVW( eye_model, ce, width, height, eye, U, V, W );  

    View view = {} ; 
    view.update(eye_model, ce, width, height) ; 

    assert( view.eye.x == eye.x );  
    assert( view.eye.y == eye.y );  
    assert( view.eye.z == eye.z );  

    assert( view.U.x == U.x );  
    assert( view.U.y == U.y );  
    assert( view.U.z == U.z );  

    assert( view.V.x == V.x );  
    assert( view.V.y == V.y );  
    assert( view.V.z == V.z );  

    assert( view.W.x == W.x );  
    assert( view.W.y == W.y );  
    assert( view.W.z == W.z );  

    view.dump("View::dump"); 
    view.save("/tmp");  


    const char* dir = "/tmp/ScanTest_scans" ; 

    const Solid* solid0 = foundry.getSolid(0); 

    Scan scan(dir, &foundry, solid0 ); 
    scan.axis_scan() ; 
    scan.rectangle_scan() ; 
    scan.circle_scan() ; 

    return 0 ;  
}
