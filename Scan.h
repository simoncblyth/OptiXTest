#pragma once

#include <string>
#include <vector>
#include "Quad.h"

struct Solid ; 

struct Scan
{
    Scan( const Solid* solid );   

    void trace(const float t_min, const float3& ray_origin, const float3& ray_direction );
    void trace(const float t_min, const float3& ray_origin, const std::vector<float3>& dirs );

    void circle_scan(); 
    void axis_scan(); 
    void rectangle_scan(); 
    void _rectangle_scan(float t_min, unsigned n, float halfside, float y ) ;

    std::string brief() const ;
    void dump( const quad4& rec ); 
    void save(const char* sub);


    const Solid* solid ; 
    std::vector<quad4> recs ; 
};



