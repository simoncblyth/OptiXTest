#pragma once
#include "Quad.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define NODE_METHOD __device__
#else
   #define NODE_METHOD 
#endif 

/**
Node (synonymous with Part)
==============================

NB elements are used for different purposes depending on typecode, 
eg planeIdx, planeNum are used only with CSG_CONVEXPOLYHEDRON. 

**/

struct Node
{
    quad q0 ;
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

    NODE_METHOD unsigned gtransformIdx() const { return q3.u.w & 0x7fffffff ; }  //  gtransformIdx is 1-based, 0 meaning None 
    NODE_METHOD bool        complement() const { return q3.u.w & 0x80000000 ; } 

    // only relevant for CSG_CONVEXPOLYHEDRON 
    NODE_METHOD unsigned planeIdx()      const { return q0.u.x ; }  // 1-based, 0 meaning None
    NODE_METHOD unsigned planeNum()      const { return q0.u.y ; } 
    NODE_METHOD void setPlaneIdx(unsigned idx){  q0.u.x = idx ; } 
    NODE_METHOD void setPlaneNum(unsigned num){  q0.u.y = num ; }


    NODE_METHOD unsigned index()     const {      return q1.u.w ; }  //  
    NODE_METHOD unsigned boundary()  const {      return q1.u.z ; }  //   see ggeo-/GPmt
    NODE_METHOD unsigned typecode()  const {      return q2.u.w ; }  //  OptickCSG_t enum 
    NODE_METHOD void setTypecode(unsigned tc){  q2.u.w = tc ; }


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
    static void Dump(const Node* n, unsigned ni, const char* label);  
#endif

};



