#pragma once

#include "Quad.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define PRIM_METHOD __device__
#else
   #define PRIM_METHOD 
#endif 

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "PrimSpec.h"
#endif


struct Prim  
{
    quad q0 ; 
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

    PRIM_METHOD int  numNode() const    { return q0.i.x ; } 
    PRIM_METHOD int  nodeOffset() const { return q0.i.y ; } 
    PRIM_METHOD int  tranOffset() const { return q0.i.z ; }
    PRIM_METHOD int  planOffset() const { return q0.i.w ; }

    PRIM_METHOD void setNumNode(   int numNode){    q0.i.x = numNode ; }
    PRIM_METHOD void setNodeOffset(int nodeOffset){ q0.i.y = nodeOffset ; }
    PRIM_METHOD void setTranOffset(int tranOffset){ q0.i.z = tranOffset ; }
    PRIM_METHOD void setPlanOffset(int planOffset){ q0.i.z = planOffset ; }

    PRIM_METHOD unsigned  sbtIndexOffset()    const { return  q1.u.x ; }
    PRIM_METHOD void  setSbtIndexOffset(unsigned sbtIndexOffset){  q1.u.x = sbtIndexOffset ; }

    PRIM_METHOD const unsigned* sbtIndexOffsetPtr() const { return &q1.u.x ; }



    PRIM_METHOD void setAABB(  float e  ){                                                   q2.f.x = -e ; q2.f.y = -e ; q2.f.z = -e ; q2.f.w =  e ; q3.f.x =  e ; q3.f.y =  e ; }  
    PRIM_METHOD void setAABB(  float x0, float y0, float z0, float x1, float y1, float z1){  q2.f.x = x0 ; q2.f.y = y0 ; q2.f.z = z0 ; q2.f.w = x1 ; q3.f.x = y1 ; q3.f.y = z1 ; }  
    PRIM_METHOD const float* AABB() const {  return &q2.f.x ; }
    PRIM_METHOD const float3 mn() const {    return make_float3(q2.f.x, q2.f.y, q2.f.z) ; }
    PRIM_METHOD const float3 mx() const {    return make_float3(q2.f.w, q3.f.x, q3.f.y) ; }


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
    static PrimSpec MakeSpec( const Prim* prim, unsigned primIdx, unsigned numPrim ) ; 
#endif

};


