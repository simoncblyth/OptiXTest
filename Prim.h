#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#endif

struct Prim
{
    int numNode    ; 
    int nodeOffset ; 
    int tranOffset ; 
    int planOffset ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
#endif

};


