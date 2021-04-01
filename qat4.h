#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
   #define QAT4_METHOD __device__ __host__ __forceinline__
#else
   #define QAT4_METHOD 
#endif 


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
   #include <iostream>
   #include <iomanip>
#endif 



#include "Quad.h"

struct qat4 
{
    quad q0, q1, q2, q3 ; 

    QAT4_METHOD qat4(const float* v) 
    {
        q0.f.x = *(v+0)  ;  q0.f.y = *(v+1)  ;   q0.f.z = *(v+2)  ;  q0.f.w = *(v+3) ;   
        q1.f.x = *(v+4)  ;  q1.f.y = *(v+5)  ;   q1.f.z = *(v+6)  ;  q1.f.w = *(v+7) ;   
        q2.f.x = *(v+8)  ;  q2.f.y = *(v+9)  ;   q2.f.z = *(v+10) ;  q2.f.w = *(v+11) ;   
        q3.f.x = *(v+12) ;  q3.f.y = *(v+13) ;   q3.f.z = *(v+14) ;  q3.f.w = *(v+15) ;   
    } 

    QAT4_METHOD qat4() 
    {
        q0.f.x = 1.f ;  q0.f.y = 0.f ;   q0.f.z = 0.f ;  q0.f.w = 0.f ;   
        q1.f.x = 0.f ;  q1.f.y = 1.f ;   q1.f.z = 0.f ;  q1.f.w = 0.f ;   
        q2.f.x = 0.f ;  q2.f.y = 0.f ;   q2.f.z = 1.f ;  q2.f.w = 0.f ;   
        q3.f.x = 0.f ;  q3.f.y = 0.f ;   q3.f.z = 0.f ;  q3.f.w = 1.f ;   
    } 

    // prepare for matrix multiply by clearing any auxiliary info in the "spare" 4th column 
    QAT4_METHOD void prep()   
    {
        q0.f.w = 0.f ; 
        q1.f.w = 0.f ; 
        q2.f.w = 0.f ; 
        q3.f.w = 1.f ; 
    }


    QAT4_METHOD float* data() 
    {
        return &q0.f.x ;
    }

    QAT4_METHOD float4 operator*(const float4 &v) const    
    {
        float4 ret;
        ret.x = q0.f.x * v.x + q1.f.x * v.y + q2.f.x * v.z + q3.f.x * v.w;
        ret.y = q0.f.y * v.x + q1.f.y * v.y + q2.f.y * v.z + q3.f.y * v.w;
        ret.z = q0.f.z * v.x + q1.f.z * v.y + q2.f.z * v.z + q3.f.z * v.w;
        ret.w = q0.f.w * v.x + q1.f.w * v.y + q2.f.w * v.z + q3.f.w * v.w;
        return ret;
    }
    //  one-or-other convention here  
    // TODO : make sure are using same convention as glm 

}; 
   


#if defined(__CUDACC__) || defined(__CUDABE__)
#else


inline std::ostream& operator<<(std::ostream& os, const float3& v)
{
    int w = 10 ; 
    os 
       << "(" 
       << std::setw(w) << v.x 
       << "," 
       << std::setw(w) << v.y
       << "," 
       << std::setw(w) << v.z 
       << ") "  
       ;
    return os; 
}

inline std::ostream& operator<<(std::ostream& os, const float4& v)
{
    int w = 10 ; 
    os 
       << "(" 
       << std::setw(w) << v.x 
       << "," 
       << std::setw(w) << v.y
       << "," 
       << std::setw(w) << v.z 
       << "," 
       << std::setw(w) << v.w 
       << ") "  
       ;
    return os; 
}

inline std::ostream& operator<<(std::ostream& os, const qat4& v)
{
    os 
       << v.q0.f  
       << v.q1.f  
       << v.q2.f  
       << v.q3.f
       ;
 
    return os; 
}

#endif 



