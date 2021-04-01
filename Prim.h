#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#endif

struct PrimSpec
{
    float*    aabb ; 
    unsigned* sbtIndexOffset ;  
    unsigned num_aabb ; 
    unsigned stride_in_bytes ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void dump(const char* msg="PrimSpec::Dump") const ; 
#endif
};

struct Prim   // (3*4)
{
    float3 mn ; 
    float3 mx ; 
    unsigned sbtIndexOffset ; 
    float  pad1 ; 

    int    numNode    ;  // nodes in the tree 
    int    nodeOffset ;  // pointer to root node 
    int    tranOffset ; 
    int    planOffset ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
    static PrimSpec MakeSpec( const Prim* prim, unsigned primIdx, unsigned numPrim ) ; 
#endif

};


