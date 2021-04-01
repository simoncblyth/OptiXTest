
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
      << " mn " << mn 
      << " mx " << mx 
      << " sbtIndexOffset " << sbtIndexOffset 
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
    ps.num_prim = numPrim ; 
    ps.stride_in_bytes = sizeof(Prim); 
    return ps ; 
}

void PrimSpec::gather(std::vector<float>& out) const 
{
    assert( device == false ); 
    unsigned size_in_floats = 6 ; 
    out.resize( num_prim*size_in_floats ); 

    unsigned stride_in_floats = stride_in_bytes/sizeof(float) ; 
    for(unsigned i=0 ; i < num_prim ; i++) 
    {   
        float* dst = out.data() + size_in_floats*i ;   
        const float* src = aabb + stride_in_floats*i ;   
        memcpy(dst, src,  sizeof(float)*size_in_floats );  
    }   
}

void PrimSpec::Dump(std::vector<float>& out)  // static 
{
     std::cout << " gather " << out.size() << std::endl ; 
     for(unsigned i=0 ; i < out.size() ; i++) 
     {    
         if(i % 6 == 0) std::cout << std::endl ; 
         std::cout << std::setw(10) << out[i] << " " ; 
     } 
     std::cout << std::endl ; 
}





void PrimSpec::dump(const char* msg) const 
{
    assert( stride_in_bytes % sizeof(float) == 0 ); 
    unsigned stride_in_floats = stride_in_bytes/sizeof(float) ; 
    std::cout 
        << msg 
        << " num_prim " << num_prim 
        << " stride_in_bytes " << stride_in_bytes 
        << " stride_in_floats " << stride_in_floats 
        << std::endl 
        ; 

    for(unsigned i=0 ; i < num_prim ; i++)
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


