
#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include "sutil_vec_math.h"
#include "qat4.h"
#include "Prim.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>


std::string Prim::desc() const 
{  
    std::stringstream ss ; 
    ss 
      << "Prim"
      << " mn " << mn 
      << " mx " << mx 
      << " numNode "    << std::setw(3) << numNode 
      << " nodeOffset " << std::setw(3) << nodeOffset
      << " tranOffset " << std::setw(3) << tranOffset
      << " planOffset " << std::setw(3) << planOffset
      ;
    
    std::string s = ss.str(); 
    return s ; 
}


PrimSpec Prim::MakeSpec( const Prim* prim,  unsigned primIdx, unsigned numPrim ) // static 
{
    const Prim* pr = prim + primIdx ; 
    size_t offset_sbtIndexOffset = offsetof(struct Prim, sbtIndexOffset)/sizeof(unsigned)  ; 
    assert( offset_sbtIndexOffset == 6 ) ; 

    PrimSpec ps ; 
    ps.aabb = (float*)pr ; 
    ps.sbtIndexOffset = (unsigned*)(pr) + offset_sbtIndexOffset ;  
    ps.num_aabb = numPrim ; 
    ps.stride_in_bytes = sizeof(Prim); 
    return ps ; 
}


void PrimSpec::dump(const char* msg) const 
{
    assert( stride_in_bytes % sizeof(float) == 0 ); 
    unsigned stride_in_floats = stride_in_bytes/sizeof(float) ; 
    std::cout 
        << msg 
        << " num_aabb " << num_aabb 
        << " stride_in_bytes " << stride_in_bytes 
        << " stride_in_floats " << stride_in_floats 
        << std::endl 
        ; 

    for(unsigned i=0 ; i < num_aabb ; i++)
    {   
        std::cout 
            << " i " << std::setw(4) << i 
            << " sbtIndexOffset " << std::setw(4) << *(sbtIndexOffset + i*stride_in_floats)   
            ; 
        for(unsigned j=0 ; j < 6 ; j++)  
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << *(aabb + i*stride_in_floats + j ) << " "  ;   
        std::cout << std::endl ; 
    }   
}

#endif


