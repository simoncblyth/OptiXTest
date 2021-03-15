// name=intersect_node ; gcc $name.cc -std=c++11 -lstdc++ -I. -I/usr/local/cuda/include -o /tmp/$name && /tmp/$name

#include <vector>
#include <iostream>
#include <iomanip>
#include "sutil_vec_math.h"
#include "intersect_node.h"

void dump(const float4& isect, const bool valid_isect, const float3& ray_direction, const char* label)
{
    std::cout 
        << std::setw(30) << label
        << " valid_isect " << valid_isect 
        << " isect ( "
        << std::setw(10) << std::fixed << std::setprecision(3) << isect.x 
        << std::setw(10) << std::fixed << std::setprecision(3) << isect.y
        << std::setw(10) << std::fixed << std::setprecision(3) << isect.z 
        << std::setw(10) << std::fixed << std::setprecision(3) << isect.w
        << " ) "
        << " dir ( "
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_direction.x 
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_direction.y
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_direction.z 
        << " ) "
        << std::endl 
        ; 
}

void test_intersect_node(const Node* node, const float t_min, const float3& ray_origin, const std::vector<float3>& dirs, const char* label)
{
    for(unsigned i=0 ; i < dirs.size() ; i++)
    {
        const float3& ray_direction = dirs[i] ; 
        float4 isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ; 
        bool valid_isect = intersect_node( isect, node, t_min, ray_origin, ray_direction ); 
        dump(isect, valid_isect, ray_direction, label); 
    }
}

void test_sphere(const float t_min, const float3& ray_origin, const std::vector<float3>& dirs)
{
    Node nd ; 
    nd.q0.f = {0.f, 0.f, 0.f, 100.f } ; 
    nd.q1.u = {0,0,0,0} ; 
    nd.q2.u = {0,0,0,CSG_SPHERE} ; 
    nd.q3.u = {0,0,0,0} ; 

    test_intersect_node( &nd, t_min, ray_origin, dirs, "test_sphere" ); 
}

void test_zsphere(const float t_min, const float3& ray_origin, const std::vector<float3>& dirs)
{
    Node nd ; 
    nd.q0.f = {0.f,   0.f, 0.f, 100.f } ; 
    nd.q1.f = {-20.f,20.f, 0.f,   0.f } ; 
    nd.q2.u = {0,0,0,CSG_ZSPHERE} ; 
    nd.q3.u = {0,0,0,0} ; 

    test_intersect_node( &nd, t_min, ray_origin, dirs, "test_zsphere" ); 
}

int main(int argc, char** argv)
{
    float t_min = 0.f ;
    float3 origin = make_float3( 0.f, 0.f, 0.f ); 

    std::vector<float3> dirs ; 
    dirs.push_back( make_float3( 1.f, 0.f, 0.f));
    dirs.push_back( make_float3(-1.f, 0.f, 0.f));
    dirs.push_back( make_float3( 0.f, 1.f, 0.f));
    dirs.push_back( make_float3( 0.f,-1.f, 0.f));
    dirs.push_back( make_float3( 0.f, 0.f, 1.f));
    dirs.push_back( make_float3( 0.f, 0.f,-1.f));

    test_sphere( t_min, origin, dirs);     
    test_zsphere(t_min, origin, dirs);     

    return 0 ;  
}
