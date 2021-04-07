#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   #include <iostream>
   #include <iomanip>
#endif 

#define AABB_METHOD inline 


struct AABB
{
    float3 mn ; 
    float3 mx ; 

    static AABB Make(const float* v ); 
    const float* data() const ; 
    float3 center() const ; 
    float  extent() const ; 
    float4 center_extent() const ;     

    bool empty() const ; 
    void include_point(const float* point); 
    void include_aabb( const float* aabb);

}; 


AABB_METHOD AABB AABB::Make( const float* v )
{
    AABB bb = {} ; 
    bb.mn.x = *(v+0);  
    bb.mn.y = *(v+1);  
    bb.mn.z = *(v+2);
    bb.mx.x = *(v+3);  
    bb.mx.y = *(v+4);  
    bb.mx.z = *(v+5);
    return bb ; 
}

AABB_METHOD const float* AABB::data() const 
{
    return (const float*)&mn ;     // hmm assumes compiler adds no padding between mn and mx 
}
AABB_METHOD float3 AABB::center() const 
{
    return ( mx + mn )/2.f ;  
}
AABB_METHOD float AABB::extent() const 
{   
    float3 d = mx - mn ; 
    return fmaxf(fmaxf(d.x, d.y), d.z) /2.f ; 
}   
AABB_METHOD float4 AABB::center_extent() const 
{
    return make_float4( center(), extent() ); 
}

AABB_METHOD bool AABB::empty() const 
{   
    return mn.x == 0.f && mn.y == 0.f && mn.z == 0.f && mx.x == 0.f && mx.y == 0.f && mx.z == 0.f  ;   
}   

/*
AABB::include_point
--------------------


      +-  - - - -*   <--- included point pushing out the max, leaves min unchanged
      .          | 
      +-------+  .
      |       |  |
      |       |  .
      |       |  |
      +-------+- +

      +-------+  
      |    *  |     <-- interior point doesnt change min/max  
      |       |  
      |       |  
      +-------+ 

      +-------+-->--+  
      |       |     |  
      |       |     *  <--- side point pushes out max, leaves min unchanged
      |       |     |
      +-------+-----+ 

*/

AABB_METHOD void AABB::include_point(const float* point)
{
    const float3 p = make_float3( *(point+0), *(point+1), *(point+2) ); 

    if(empty())
    {
        mn = p ; 
        mx = p ; 
    } 
    else
    {
        mn = fminf( mn, p );
        mx = fmaxf( mx, p );
    }
}

AABB_METHOD void AABB::include_aabb(const float* aabb)
{
    const float3 other_mn = make_float3( *(aabb+0), *(aabb+1), *(aabb+2) ); 
    const float3 other_mx = make_float3( *(aabb+3), *(aabb+4), *(aabb+5) ); 

    if(empty())
    {
        mn = other_mn ; 
        mx = other_mx ; 
    } 
    else
    {
        mn = fminf( mn, other_mn );
        mx = fmaxf( mx, other_mx );
    }
}



#if defined(__CUDACC__) || defined(__CUDABE__)
#else

inline std::ostream& operator<<(std::ostream& os, const AABB& bb)
{
    os 
       << " [ "
       << bb.mn
       << " : "
       << bb.mx 
       << " | "
       << ( bb.mx - bb.mn )
       << " ] "
       ;
    return os; 
}
#endif 


