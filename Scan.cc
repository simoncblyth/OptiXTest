
#include "sutil_vec_math.h"
#include "intersect_node.h"

#include "NP.hh"
#include "Solid.h"
#include "Scan.h"

Scan::Scan( const Solid* solid_ ) 
    :
    solid(solid_)
{
}

void Scan::trace(const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    float4 isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ; 
    bool valid_isect = intersect_node( isect, &solid->prim, &solid->node, solid->plan.data() , t_min, ray_origin, ray_direction ); 


    quad4 rec ;  
    rec.q0.f = make_float4(ray_origin); 
    rec.q0.i.w = int(valid_isect) ; 

    rec.q1.f = make_float4(ray_direction); 
    rec.q2.f = make_float4(0.f); 
    rec.q3.f = isect ; 
    //rec.q3.f = make_int4( valid_isect, 0, 0, 0);         

    if(valid_isect)
    {
        float t = isect.w ; 
        float3 pos = ray_origin + t*ray_direction ; 
        rec.q2.f = make_float4(pos, t ); 
    }
    recs.push_back(rec);  

    dump(rec); 
}


void Scan::dump( const quad4& rec )  // stat
{
    bool valid_isect = rec.q0.i.w == 1 ; 

    const float4& isect = rec.q3.f ; 
    const float4& ray_origin  = rec.q0.f ; 
    const float4& ray_direction = rec.q1.f ; 

    std::cout 
        << std::setw(30) << solid->label
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
        << " ori ( "
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.x 
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.y
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.z 
        << " ) "
        << std::endl 
        ; 

}

std::string Scan::brief() const
{
    unsigned nhit = 0 ; 
    unsigned nmiss = 0 ; 

    for(unsigned i=0 ; i < recs.size() ; i++)
    {
        const quad4& rec = recs[i] ; 
        bool hit = rec.q0.i.w == 1 ; 
        if(hit)  nhit += 1 ; 
        if(!hit) nmiss += 1 ; 
    }
    std::stringstream ss ; 
    ss
        << " nhit " << nhit 
        << " nmiss " << nmiss 
        ;

    std::string s = ss.str() ; 
    return s ; 
}


void Scan::trace(const float t_min, const float3& ray_origin, const std::vector<float3>& dirs )
{
    for(unsigned i=0 ; i < dirs.size() ; i++)
    {
        const float3& ray_direction = dirs[i] ; 
        trace( t_min, ray_origin, ray_direction ); 
    }
}

void Scan::circle_scan()
{
    float t_min = 0.f ;
    float radius = 2.0f*solid->extent ; 

    // M_PIf from sutil_vec_math.h
    for(float phi=0. ; phi <= M_PIf*2.0 ; phi+=M_PIf*2.0/1000.0 )
    {
        float3 origin = make_float3( radius*sin(phi), 0.f, radius*cos(phi) ); 
        float3 direction = make_float3( -sin(phi),  0.f, -cos(phi) ); 
        trace(t_min, origin, direction );     
    }
    save("circle_scan");  
}


void Scan::_rectangle_scan(float t_min, unsigned n, float halfside, float y )
{
    // shooting up/down 

    float3 z_up   = make_float3( 0.f, 0.f,  1.f);
    float3 z_down = make_float3( 0.f, 0.f, -1.f);

    float3 z_top = make_float3( 0.f, y,  halfside ); 
    float3 z_bot = make_float3( 0.f, y, -halfside ); 

    // shooting left/right

    float3 x_right = make_float3(  1.f, 0.f,  0.f);
    float3 x_left  = make_float3( -1.f, 0.f,  0.f);

    float3 x_lhs = make_float3( -halfside, y,  0.f ); 
    float3 x_rhs = make_float3(  halfside, y,  0.f ); 

    for(float v=-halfside ; v <= halfside ; v+= halfside/float(n) )
    { 
        z_top.x = v ; 
        z_bot.x = v ; 

        trace(t_min, z_top, z_down );     
        trace(t_min, z_bot, z_up   );     

        x_lhs.z = v ; 
        x_rhs.z = v ; 
        trace(t_min, x_lhs, x_right );     
        trace(t_min, x_rhs, x_left  );     
    }
}

void Scan::rectangle_scan()
{
    float halfside = 2.0f*solid->extent ; 
    unsigned nxz = 100 ; 
    unsigned ny = 10 ; 
    float t_min = 0.f ;

    for(float y=-halfside ; y <= halfside ; y += halfside/float(ny) )
    {
        _rectangle_scan( t_min, nxz, halfside,   y );  
    }
    save("rectangle_scan"); 
}

void Scan::axis_scan()
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

    trace(t_min, origin, dirs );     

    save("axis_scan"); 
}

void Scan::save(const char* sub)
{
    std::cout << " recs.size " << recs.size() << std::endl ; 
    if(recs.size() == 0 ) return ; 

    std::string name = solid->label ; 
    name += ".npy" ; 

    std::stringstream ss ; 
    ss << "/tmp/intersect_node_tests/" << sub ; 
    std::string dir = ss.str(); 

    NP::Write( dir.c_str(), name.c_str(), (float*)recs.data(), recs.size(), 4, 4 ) ; 
    recs.clear(); 
}



