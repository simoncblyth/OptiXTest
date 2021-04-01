#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <vector>
#include <string>
#endif

struct PrimSpec
{
    float*    aabb ; 
    unsigned* sbtIndexOffset ;  
    unsigned  num_prim ; 
    unsigned  stride_in_bytes ; 
    bool      device ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    void gather(std::vector<float>& out) const ;
    static void Dump(std::vector<float>& out);
    void dump(const char* msg="PrimSpec::Dump") const ; 
#endif
};

#if defined(__CUDACC__) // NVCC
   #define MY_ALIGN(n) __align__(n)
#elif defined(__GNUC__) // GCC
  #define MY_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
  #define MY_ALIGN(n) __declspec(align(n))
#else
  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
#endif


struct __align__(8) Prim   // (3*4)
{
    float3   mn ; 
    float3   mx ; 
    unsigned sbtIndexOffset ; 
    float    pad1 ; 

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


