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
eg planeIdx, planeNum are used only with CSG_CONVEXPOLYHEDRON.  Marked "cx:" below.

* vim replace : shift-R


    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    | q  |      x         |      y         |     z          |      w         |  notes                                          |
    +====+================+================+================+================+=================================================+
    |    | sp/zs/cy:cen_x | sp/zs/cy:cen_y | sp/zs/cy:cen_z | sp/zs/cy:radius|  eliminate center? as can be done by transform  |
    | q0 | cn:r1          | cn:z1          | cn:r2          | cn:z2          |  cn:z2 > z1                                     |
    |    | hy:r0 z=0 waist| hy:zf          | hy:z1          | hy:z2          |  hy:z2 > z1                                     |
    |    | b3:fx          | b3:fy          | b3:fz          |                |  b3: fullside dimensions, center always origin  |
    |    | pl/sl:nx       | pl/sl:ny       | pl/sl:nz       | pl:d           |  pl: NB Node plane distinct from plane array    |
    |    |                |                | ds:inner_r     | ds:radius      |                                                 |
    |    |                |                |                |                |                                                 |
    |    | cx:planeIdx    | cx:planeNum    |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    | zs:zdelta_0    | zs:zdelta_1    | boundary       | index          |                                                 |
    |    | sl:a           | sl:b           |                |                |  sl:a,b offsets from origin                     |
    | q1 | cy:z1          | cy:z2          |                |                |  cy:z2 > z1                                     |
    |    | ds:z1          | ds:z2          |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                |                |(prev:typecode) |                                                 |
    |    |                |                |                |                |                                                 |
    | q2 |  BBMin_x       |  BBMin_y       |  BBMin_z       |  BBMax_x       |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+
    |    |                |                | (typecode)     | gtransformIdx  |                                                 |
    |    |                |                |                | complement     |                                                 |
    | q3 |  BBMax_y       |  BBMax_z       |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    |    |                |                |                |                |                                                 |
    +----+----------------+----------------+----------------+----------------+-------------------------------------------------+


* moving typecode would give 6 contiguous slots for aabb

**/

struct Node
{
    quad q0 ;
    quad q1 ; 
    quad q2 ; 
    quad q3 ; 

    NODE_METHOD unsigned gtransformIdx() const { return q3.u.w & 0x7fffffff ; }  //  gtransformIdx is 1-based, 0 meaning None 
    NODE_METHOD bool        complement() const { return q3.u.w & 0x80000000 ; } 

    NODE_METHOD unsigned planeIdx()      const { return q0.u.x ; }  // 1-based, 0 meaning None
    NODE_METHOD unsigned planeNum()      const { return q0.u.y ; } 
    NODE_METHOD void setPlaneIdx(unsigned idx){  q0.u.x = idx ; } 
    NODE_METHOD void setPlaneNum(unsigned num){  q0.u.y = num ; }

    NODE_METHOD void setTransform(  unsigned idx ){     q3.u.w |= (idx & 0x7fffffff) ; }
    NODE_METHOD void setComplement( bool complement ){  q3.u.w |= ( (int(complement) << 31) & 0x80000000) ; }

    NODE_METHOD void setParam( float x , float y , float z , float w , float z1, float z2 ){ q0.f.x = x  ; q0.f.y = y  ; q0.f.z = z  ; q0.f.w = w  ; q1.f.x = z1 ; q1.f.y = z2 ;  }
    NODE_METHOD void setAABB(  float x0, float y0, float z0, float x1, float y1, float z1){  q2.f.x = x0 ; q2.f.y = y0 ; q2.f.z = z0 ; q2.f.w = x1 ; q3.f.x = y1 ; q3.f.y = z1 ; }  
    NODE_METHOD void setAABB(  float e ){                                                    q2.f.x = -e ; q2.f.y = -e ; q2.f.z = -e ; q2.f.w =  e ; q3.f.x =  e ; q3.f.y =  e ; }  

    NODE_METHOD       float* AABB()       {  return &q2.f.x ; }
    NODE_METHOD const float* AABB() const {  return &q2.f.x ; }
    NODE_METHOD const float3 mn() const {    return make_float3(q2.f.x, q2.f.y, q2.f.z) ; }
    NODE_METHOD const float3 mx() const {    return make_float3(q2.f.w, q3.f.x, q3.f.y) ; }
    NODE_METHOD float extent() const 
    {
        float3 d = make_float3( q2.f.w - q2.f.x, q3.f.x - q2.f.y, q3.f.y - q2.f.z ); 
        return fmaxf(fmaxf(d.x, d.y), d.z) /2.f ; 
    }



    NODE_METHOD unsigned index()     const {      return q1.u.w ; }  //  
    NODE_METHOD unsigned boundary()  const {      return q1.u.z ; }  //   see ggeo-/GPmt

    NODE_METHOD unsigned typecode()  const {      return q3.u.z ; }  //  OptickCSG_t enum 
    NODE_METHOD void setTypecode(unsigned tc){           q3.u.z = tc ; }




#if defined(__CUDACC__) || defined(__CUDABE__)
#else
    std::string desc() const ; 
    static void Dump(const Node* n, unsigned ni, const char* label);  

    static const float UNBOUNDED_DEFAULT_EXTENT ; 

    static Node Union(); 
    static Node Intersection(); 
    static Node Difference(); 
    static Node BooleanOperator(char op); 

    static Node Sphere(float radius);
    static Node ZSphere(float radius, float z1, float z2);
    static Node Cone(float r1, float z1, float r2, float z2); 
    static Node Hyperboloid(float r0, float zf, float z1, float z2);
    static Node Box3(float fx, float fy, float fz ); 
    static Node Box3(float fullside); 
    static Node Plane(float nx, float ny, float nz, float d);
    static Node Slab(float nx, float ny, float nz, float d1, float d2 ) ;
    static Node Cylinder(float px, float py, float radius, float z1, float z2) ;
    static Node Disc(float px, float py, float ir, float r, float z1, float z2);

    static Node Make(const char* name); 

#endif

};


