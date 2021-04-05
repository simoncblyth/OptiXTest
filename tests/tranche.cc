//  name=tranche ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <vector>
#include <iostream>
#include <iomanip>

#include "error.h"
#include "tranche.h"

void dump(unsigned long long bef, unsigned long long aft, unsigned slice, float tmin, int ierr)
{
       std::cout 
            << " bef: " << std::hex << std::setw(16) << bef   << std::dec
            << " sli: " << std::hex << std::setw(4)  << slice << std::dec 
            << " aft: " << std::hex << std::setw(16) << aft   << std::dec
            << " tmin: " << std::setw(10) << tmin 
            << " ierr: " << ierr 
            << std::endl
            ; 
}

int main(int argc, char** argv)
{
    unsigned slice ; 
    float tmin ; 
    int ierr = 0 ; 
    unsigned long long bef, aft ; 

    Tranche tr ; 
    tr.curr = -1 ; 

    std::vector<unsigned> slices = { 0xaaaa, 0xbbbb, 0xcccc, 0xdddd } ; 
    std::vector<float>    tmins  = { 0.1f, 1.1f , 2.1f, 3.1f } ; 

    for(unsigned i=0 ; i < slices.size() ; i++)
    {
        slice = slices[i] ;  
        tmin = tmins[i] ; 

        bef = tranche_repr(tr) ; 
        ierr = tranche_push( tr, slice  ,  tmin );
        aft = tranche_repr(tr) ; 
        dump( bef, aft, slice, tmin, ierr ); 
    }

    while (tr.curr > -1)
    {    
        bef = tranche_repr(tr) ; 
        ierr = tranche_pop(tr, slice, tmin );
        aft = tranche_repr(tr) ; 
        dump(bef, aft, slice, tmin, ierr ); 
 
        if(ierr) break ; 
    }
    
    std::cout << " ierr " << ierr << std::endl ; 
    return 0 ; 
}

