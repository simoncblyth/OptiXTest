// ./csg.sh 


#include <vector>
#include <iostream>
#include <iomanip>
#include "sutil_vec_math.h"

#include "error.h"
#include "csg.h"


void dump(unsigned long long bef, unsigned long long aft, const float4& isect, unsigned nodeIdx, int ierr, const char* msg )
{
       std::cout 
            << " bef: " << std::hex << std::setw(16) << bef   << std::dec
            << " nidx: " << std::hex << std::setw(4)  << nodeIdx << std::dec 
            << " aft: " << std::hex << std::setw(16) << aft   << std::dec
            << " isect: (" 
            << " " << std::setw(10) << std::setprecision(4) << std::fixed << isect.x 
            << " " << std::setw(10) << std::setprecision(4) << std::fixed << isect.y
            << " " << std::setw(10) << std::setprecision(4) << std::fixed << isect.z 
            << " " << std::setw(10) << std::setprecision(4) << std::fixed << isect.w 
            << ")" 
            << " ierr: " << ierr 
            << " " << msg 
            << std::endl
            ; 
}



int main(int argc, char** argv)
{
    int ierr = 0 ; 
    unsigned long long bef, aft ; 

    std::vector<unsigned> nodeIdxs = { 0xaaaa, 0xbbbb, 0xcccc, 0xdddd } ; 
    std::vector<float4>   isects  = { 
                                       { 1.1f, 1.1f , 1.1f, 1.1f}, 
                                       { 2.1f, 2.1f , 2.1f, 2.1f}, 
                                       { 3.1f, 3.1f , 3.1f, 3.1f}, 
                                       { 4.1f, 4.1f , 4.1f, 4.1f}, 
                                    } ; 

    float4 isect ; 
    unsigned nodeIdx ; 

    CSG_Stack csg ;  
    csg.curr = -1 ; 

    for(unsigned i=0 ; i < isects.size() ; i++)
    {
        isect = isects[i] ;  
        nodeIdx = nodeIdxs[i] ; 

        bef = csg_repr(csg) ; 
        ierr = csg_push( csg, isect,  nodeIdx );
        aft = csg_repr(csg) ; 
        dump( bef, aft, isect, nodeIdx, ierr, "push" ); 
    }

    while (csg.curr > -1)
    {    
        bef = csg_repr(csg) ; 
        ierr = csg_pop(csg, isect, nodeIdx );
        aft = csg_repr(csg) ; 
        dump(bef, aft, isect, nodeIdx,  ierr, "pop" ); 
 
        if(ierr) break ; 
    }
    std::cout << " ierr " << ierr << std::endl ; 
    return 0 ; 
}
