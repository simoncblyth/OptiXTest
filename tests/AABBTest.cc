#include <cassert>
#include <iostream>

#include "sutil_vec_math.h"
#include "AABB.h"

int main(int argc, char** argv)
{
    AABB bb = {} ; 
    std::cout << " bb0 " << bb << std::endl ; 
    assert( bb.empty() ); 

    bb = {-100.f, -100.f, -100.f,  100.f, 100.f, 100.f } ; 
    std::cout << " bb1 " << bb << std::endl ; 
    assert( !bb.empty() ); 

    float3 p = make_float3( 200.f, 200.f, 300.f ); 
    bb.include_point( (const float*)&p ) ;
    std::cout << " bb2 " << bb << std::endl ; 

    AABB other_1 = { -50.f, -50.f, -50.f, 50.f, 50.f, 50.f } ; 
    bb.include_aabb( (const float*)&other_1 ) ;
    std::cout << " bb3 " << bb << std::endl ; 

    AABB other_2 = { -500.f, -500.f, -500.f, 500.f, 500.f, 500.f } ; 
    bb.include_aabb( (const float*)&other_2 ) ;
    std::cout << " bb4 " << bb << std::endl ; 


    std::cout << " bb.center_extent() " << bb.center_extent() << std::endl ; 


    return 0 ; 
}
