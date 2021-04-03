#pragma once

#include <string>
#include <vector>
#include "Quad.h"

struct Foundry ; 
struct Solid ; 

struct Scan
{
    Scan( const char* dir_, const Foundry* foundry_, const Solid* solid_ );   

    void trace(const float t_min, const float3& ray_origin, const float3& ray_direction );
    void trace(const float t_min, const float3& ray_origin, const std::vector<float3>& dirs );

    void record(bool valid_isect, const float4& isect,  const float3& ray_origin, const float3& ray_direction ) ;

    void circle_scan(); 
    void axis_scan(); 
    void rectangle_scan(); 
    void _rectangle_scan(float t_min, unsigned n, float halfside, float y ) ;

    std::string brief() const ;
    void dump( const quad4& rec ); 
    void save(const char* sub);


    const char* dir ; 
    const Foundry* foundry ; 
    const Solid* solid ; 
    std::vector<quad4> recs ; 
};



