#pragma once
#include "Quad.h"

struct Node
{
    quad q0 ;
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    static void Dump(const Node* n, unsigned ni, const char* label);  
#endif

};



