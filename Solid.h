#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <string>
#endif

struct Solid   // Composite shape 
{
    char        label[4] ; 
    int         numPrim ; 
    int         primOffset ;
    int         padding ;   // TODO: move to label[8] instead of this padding 

    float4      center_extent ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
#endif

};


