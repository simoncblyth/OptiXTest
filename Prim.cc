
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
#include <cstring>


std::string Prim::desc() const 
{  
    std::stringstream ss ; 
    ss 
      << "Prim"
      << " mn " << mn() 
      << " mx " << mx() 
      << " sbtIndexOffset " << sbtIndexOffset() 
      << " numNode "    << std::setw(3) << numNode() 
      << " nodeOffset " << std::setw(3) << nodeOffset()
      << " tranOffset " << std::setw(3) << tranOffset()
      << " planOffset " << std::setw(3) << planOffset()
      ;
    
    std::string s = ss.str(); 
    return s ; 
}


PrimSpec Prim::MakeSpec( const Prim* prim0,  unsigned primIdx, unsigned numPrim ) // static 
{
    const Prim* prim = prim0 + primIdx ; 

    PrimSpec ps ; 
    ps.aabb = prim->AABB() ; 
    ps.sbtIndexOffset = prim->sbtIndexOffsetPtr() ;  
    ps.num_prim = numPrim ; 
    ps.stride_in_bytes = sizeof(Prim); 
    return ps ; 
}

#endif


