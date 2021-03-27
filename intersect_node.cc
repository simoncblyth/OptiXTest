// ./intersect_node.sh

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "sutil_vec_math.h"
#include "intersect_node.h"

#include <math.h>
#include "NP.hh"


struct Solid
{
    std::string label ; 
    Prim prim ; 
    Node node ; 
    std::vector<float4> plan ; 

    static Solid* MakeSphere(); 
    static Solid* MakeZSphere(); 
    static Solid* MakeCone(); 
    static Solid* MakeConvexPolyhedron();

    static void   MakeSolids(std::vector<Solid*>& solids ) ;
};

Solid* Solid::MakeSphere()
{
    Solid* solid = new Solid ; 
    solid->label = "sphere" ; 
    solid->prim = {} ; 
    solid->node.q0.f = {0.f, 0.f, 0.f, 100.f } ; 
    solid->node.q1.u = {0,0,0,0} ; 
    solid->node.q2.u = {0,0,0,CSG_SPHERE} ; 
    solid->node.q3.u = {0,0,0,0} ; 
    return solid ; 
}
Solid* Solid::MakeZSphere()
{
    Solid* solid = new Solid ; 
    solid->label = "zsphere" ; 
    solid->prim = {} ; 
    solid->node.q0.f = {0.f,   0.f, 0.f, 100.f } ; 
    solid->node.q1.f = {-20.f,20.f, 0.f,   0.f } ; 
    solid->node.q2.u = {0,0,0,CSG_ZSPHERE} ; 
    solid->node.q3.u = {0,0,0,0} ; 
    return solid ; 
}
Solid* Solid::MakeCone()
{
     /*
     Notice problems at corners marked (*) for rays in -z direction

                       *
                       |     [0,0,0]
      0----------------A------------------           
                      / \
                     /   \
                    /     \
                 * /       \ * 
                 |/         \|
                 +-----------+         z = z2
                /     r2      \
               /               \
              /                 \
             /                   \
            +----|-----|-----------+   z = z1
      [-300,0,-300]    r1       [+300,0,-300]


     */

    Solid* solid = new Solid ; 
    solid->label = "cone" ; 
    float r2 = 100.f ;
    float r1 = 300.f ;
    float z2 = -100.f ;  
    float z1 = -300.f ;
    assert( z2 > z1 ); 
    float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex
    solid->node.q0.f = {r1, z1, r2, z2 } ; 
    solid->node.q1.f = {0.f, 0.f, 0.f, 0.f } ; 
    solid->node.q2.u = {0,0,0,CSG_CONE} ; 
    solid->node.q3.u = {0,0,0,0} ; 
    solid->prim.q0.u = {0, 0, 0, 0} ; 
    return solid ; 
}
Solid* Solid::MakeConvexPolyhedron()
{
    Solid* solid = new Solid ; 
    solid->label = "convexpolyhedron_cube" ; 
    float hx = 10.f ; 
    float hy = 20.f ; 
    float hz = 30.f ; 
    solid->plan.push_back( make_float4(  1.f,  0.f,  0.f, hx ) ); 
    solid->plan.push_back( make_float4( -1.f,  0.f,  0.f, hx ) ); 
    solid->plan.push_back( make_float4(  0.f,  1.f,  0.f, hy ) ); 
    solid->plan.push_back( make_float4(  0.f, -1.f,  0.f, hy ) ); 
    solid->plan.push_back( make_float4(  0.f,  0.f,  1.f, hz ) ); 
    solid->plan.push_back( make_float4(  0.f,  0.f, -1.f, hz ) );
    solid->node.q0.f = {0.f, 0.f, 0.f, 0.f } ; 
    solid->node.q1.f = {0.f, 0.f, 0.f, 0.f } ; 
    solid->node.q2.u = {0,0,0,CSG_CONVEXPOLYHEDRON} ; 
    solid->node.q3.u = {0,0,0,0} ; 
    solid->prim.q0.u = {0,0,0,0} ; 
    return solid ; 
}
void Solid::MakeSolids(std::vector<Solid*>& solids )
{
    solids.push_back( Solid::MakeSphere() ); 
    solids.push_back( Solid::MakeZSphere() ); 
    solids.push_back( Solid::MakeCone() ); 
    solids.push_back( Solid::MakeConvexPolyhedron() ); 
}



void dump(const float4& isect, const bool valid_isect, const float3& ray_direction, const float3& ray_origin, const char* label)
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
        << " ori ( "
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.x 
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.y
        << std::setw(10) << std::fixed << std::setprecision(3) << ray_origin.z 
        << " ) "
        << std::endl 
        ; 
}

void test_intersect(std::vector<float4>& posts, const Solid* solid, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    float4 isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ; 
    bool valid_isect = intersect_node( isect, &solid->prim, &solid->node, solid->plan.data() , t_min, ray_origin, ray_direction ); 

    dump(isect, valid_isect, ray_direction, ray_origin, solid->label.c_str() ); 

    if(valid_isect)
    {
        float4 post = make_float4( 0.f, 0.f, 0.f, 0.f ); 
        float t = isect.w ; 
        float3 pos = ray_origin + t*ray_direction ; 
        post.x = pos.x ; 
        post.y = pos.y ; 
        post.z = pos.z ; 
        post.w = t ; 

        posts.push_back(post);  
    }
}

void test_intersect(std::vector<float4>& posts, const Solid* solid, const float t_min, const float3& ray_origin, const std::vector<float3>& dirs )
{
    for(unsigned i=0 ; i < dirs.size() ; i++)
    {
        const float3& ray_direction = dirs[i] ; 
        test_intersect( posts, solid, t_min, ray_origin, ray_direction ); 
    }
}

void axis_scan(const Solid* solid)
{
    std::vector<float4> posts ; 

    float t_min = 0.f ;
    float3 origin = make_float3( 0.f, 0.f, 0.f ); 

    std::vector<float3> dirs ; 
    dirs.push_back( make_float3( 1.f, 0.f, 0.f));
    dirs.push_back( make_float3(-1.f, 0.f, 0.f));
    dirs.push_back( make_float3( 0.f, 1.f, 0.f));
    dirs.push_back( make_float3( 0.f,-1.f, 0.f));
    dirs.push_back( make_float3( 0.f, 0.f, 1.f));
    dirs.push_back( make_float3( 0.f, 0.f,-1.f));

    test_intersect(posts, solid,  t_min, origin, dirs );     

    std::string name = solid->label ; 
    name += ".npy" ; 

    std::cout << " posts.size " << posts.size() << std::endl ; 

    if(posts.size() > 0)
    {
        NP::Write("/tmp/intersect_node_tests/axis_scan", name.c_str(),  (float*)posts.data(), posts.size(), 4 ); 
        posts.clear(); 
    }
}


void x_scan(const Solid* solid)
{
    std::vector<float4> posts ; 

    float t_min = 0.f ;
    float3 direction = make_float3( 0.f, 0.f, -1.f);
    float3 origin = make_float3( 0.f, 0.f,  0.f ); 

    for(float x=-400.f ; x <= 400.f ; x+= 1.f )
    { 
        origin.x = x ; 
        test_intersect(posts,  solid,  t_min, origin, direction );     
    }

    std::string name = solid->label ; 
    name += ".npy" ; 
    std::cout << " posts.size " << posts.size() << std::endl ; 
    if(posts.size() > 0)
    {
        NP::Write("/tmp/intersect_node_tests/x_scan", name.c_str(),  (float*)posts.data(), posts.size(), 4 ); 
        posts.clear(); 
    }
}

void circle_scan(const Solid* solid)
{
    std::vector<float4> posts ; 

    float t_min = 0.f ;
    float radius = 1000.f ; 

    for(float phi=0. ; phi < M_PI*2.0 ; phi+=M_PI*2.0/100.0 )
    {
        float3 origin = make_float3( radius*sin(phi), 0.f, radius*cos(phi) ); 
        float3 direction = make_float3( -sin(phi),  0.f, -cos(phi) ); 
        test_intersect(posts,  solid,  t_min, origin, direction );     
    }

    std::string name = solid->label ; 
    name += ".npy" ; 
    std::cout << " posts.size " << posts.size() << std::endl ; 

    if(posts.size() > 0)
    {
        NP::Write("/tmp/intersect_node_tests/circle_scan", name.c_str(),  (float*)posts.data(), posts.size(), 4 ); 
        posts.clear(); 
    }
}


int main(int argc, char** argv)
{
    std::vector<Solid*> solids ; 
    Solid::MakeSolids(solids); 

    for(unsigned i=0 ; i < solids.size() ; i++)
    {
        Solid* solid = solids[i] ; 
        //axis_scan( solid ); 
        //x_scan( solid ); 
        circle_scan( solid ); 
    }

    return 0 ;  
}
