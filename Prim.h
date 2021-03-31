#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#endif

struct Prim
{
    int numNode    ; 
    int nodeOffset ; 

    // hmm : with global pools seems do not need tranOffset + planOffset anymore ? 
    // the old way of handling geometry ingredients separately for each solid adds complexity 

    int tranOffset ; 
    int planOffset ; // 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
#endif

};


