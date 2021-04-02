#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <vector>
#endif

struct PrimSpec
{
    const float*    aabb ; 
    const unsigned* sbtIndexOffset ;  
    unsigned        num_prim ; 
    unsigned        stride_in_bytes ; 
    bool            device ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void gather(std::vector<float>& out) const ;
    static void Dump(std::vector<float>& out);
    void dump(const char* msg="PrimSpec::Dump") const ; 
#endif
};

