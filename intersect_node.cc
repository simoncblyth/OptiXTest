// ./intersect_node.sh

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "sutil_vec_math.h"
#include "intersect_node.h"

#include "NP.hh"


struct Solid
{
    std::string label ; 
    Prim prim ; 
    Node node ; 
    std::vector<float4> plan ; 
    float extent ; 

    static Solid* Make(const char* name); 
    static Solid* MakeSphere(); 
    static Solid* MakeZSphere(); 
    static Solid* MakeCone(); 
    static Solid* MakeConvexPolyhedronCube();
    static Solid* MakeConvexPolyhedronTetrahedron();
    static float4 face_plane( const std::vector<float3>& v, unsigned i0, unsigned i1, unsigned i2 );
    static Solid* MakeConvexPolyhedron(std::vector<float4>& plan);
    static Solid* MakeHyperboloid();
    static Solid* MakeBox3();
    static Solid* MakePlane();
    static Solid* MakeSlab();
    static Solid* MakeCylinder();
    static Solid* MakeDisc();

    static void   MakeSolids(std::vector<Solid*>& solids ) ;
};


Solid* Solid::Make(const char* name)
{
    if(     strcmp(name, "sphere") == 0)           return MakeSphere() ;
    else if(strcmp(name, "zsphere") == 0)          return MakeZSphere() ;
    else if(strcmp(name, "cone") == 0)             return MakeCone() ;
    else if(strcmp(name, "convexpolyhedron_cube") == 0) return MakeConvexPolyhedronCube() ;
    else if(strcmp(name, "convexpolyhedron_tetrahedron") == 0) return MakeConvexPolyhedronTetrahedron() ;
    else if(strcmp(name, "hyperboloid") == 0)      return MakeHyperboloid() ;
    else if(strcmp(name, "box3") == 0)             return MakeBox3() ;
    else if(strcmp(name, "plane") == 0)            return MakePlane() ;
    else if(strcmp(name, "slab") == 0)             return MakeSlab() ;
    else if(strcmp(name, "cylinder") == 0)         return MakeCylinder() ;
    else if(strcmp(name, "disc") == 0)             return MakeDisc() ;
    else return nullptr ; 
}

Solid* Solid::MakeSphere()
{
    Solid* solid = new Solid ; 
    solid->label = "sphere" ; 
    solid->prim = {} ; 
    solid->node.q0.f = {0.f, 0.f, 0.f, 100.f } ; 
    solid->node.q1.u = {0,0,0,0} ; 
    solid->node.q2.u = {0,0,0,CSG_SPHERE} ; 
    solid->node.q3.u = {0,0,0,0} ; 
    solid->extent = 100.f ; 
    return solid ; 
}
Solid* Solid::MakeZSphere()
{
    Solid* solid = new Solid ; 
    solid->label = "zsphere" ; 
    solid->prim = {} ; 
    solid->node.q0.f = {0.f,   0.f, 0.f, 100.f } ; 
    solid->node.q1.f = {-50.f,50.f, 0.f,   0.f } ; 
    solid->node.q2.u = {0,0,0,CSG_ZSPHERE} ; 
    solid->node.q3.u = {0,0,0,0} ; 
    solid->extent = 100.f ; 
    return solid ; 
}
Solid* Solid::MakeCone()
{
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
    solid->extent = 500.f ;  // guess 
    return solid ; 
}

Solid* Solid::MakeConvexPolyhedronCube()
{
    float hx = 10.f ; 
    float hy = 20.f ; 
    float hz = 30.f ; 

    std::vector<float4> plan ; 
    plan.push_back( make_float4(  1.f,  0.f,  0.f, hx ) ); 
    plan.push_back( make_float4( -1.f,  0.f,  0.f, hx ) ); 
    plan.push_back( make_float4(  0.f,  1.f,  0.f, hy ) ); 
    plan.push_back( make_float4(  0.f, -1.f,  0.f, hy ) ); 
    plan.push_back( make_float4(  0.f,  0.f,  1.f, hz ) ); 
    plan.push_back( make_float4(  0.f,  0.f, -1.f, hz ) );

    Solid* solid = MakeConvexPolyhedron(plan); 

    solid->label = "convexpolyhedron_cube" ; 
    solid->extent = 30.f ;  
    return solid ; 
}


  /*  
     https://en.wikipedia.org/wiki/Tetrahedron

       0:(1,1,1)
       1:(1,−1,−1)
       2:(−1,1,−1) 
       3:(−1,−1,1)


                              (1,1,1)
                 +-----------0
                /|          /| 
     (-1,-1,1) / |         / |
              3-----------+  |
              |  |        |  |
              |  |        |  |
   (-1,1,-1)..|..2--------|--+
              | /         | /
              |/          |/
              +-----------1
                          (1,-1,-1)      

          Faces         (attempt to right-hand-rule orient normals outwards)
                0-1-2
                1-3-2
                3-0-2
                0-3-1


         z  y
         | /
         |/
         +---> x


         012:    x + y - z = 1

    */


float4 Solid::face_plane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k )
{
    // normal for plane through v[i] v[j] v[k]
    float3 ij = v[j] - v[i] ; 
    float3 ik = v[k] - v[i] ; 
    float3 n = normalize(cross(ij, ik )) ;
    float di = dot( n, v[i] ) ;
    float dj = dot( n, v[j] ) ;
    float dk = dot( n, v[k] ) ;
    std::cout 
        << " di " << di 
        << " dj " << dj 
        << " dk " << dk
        << " n (" << n.x << "," << n.y << "," << n.z << ")" 
        << std::endl
        ; 

    float4 plane = make_float4( n, di ) ; 
    return plane ;  
}



Solid* Solid::MakeConvexPolyhedronTetrahedron()
{
    float s = 100.f*sqrt(3) ; 

    std::vector<float3> vtx ; 
    vtx.push_back(make_float3( s, s, s));  
    vtx.push_back(make_float3( s,-s,-s)); 
    vtx.push_back(make_float3(-s, s,-s)); 
    vtx.push_back(make_float3(-s,-s, s)); 

    std::vector<float4> plan ; 
    plan.push_back(face_plane(vtx, 0, 1, 2)) ;  
    plan.push_back(face_plane(vtx, 1, 3, 2)) ;  
    plan.push_back(face_plane(vtx, 3, 0, 2)) ;  
    plan.push_back(face_plane(vtx, 0, 3, 1)) ;  

    for(unsigned i=0 ; i < plan.size() ; i++) 
    {
        const float4& p = plan[i] ;  
        std::cout << " plan (" << p.x << "," << p.y << "," << p.z << "," << p.w << ") " << std::endl ;
    } 

    Solid* solid = MakeConvexPolyhedron(plan); 
    solid->label = "convexpolyhedron_tetrahedron" ; 
    solid->extent = s ;  
    return solid ; 
}



Solid* Solid::MakeConvexPolyhedron(std::vector<float4>& plan)
{
    Solid* solid = new Solid ; 
    for(unsigned i=0 ; i < plan.size() ; i++) solid->plan.push_back( plan[i] ); 

    solid->prim = {} ; 
    solid->node = {} ;
    solid->node.setTypecode(CSG_CONVEXPOLYHEDRON) ; 
    solid->node.setPlaneNum(plan.size());    // must ve 

    return solid ; 
}

Solid* Solid::MakeHyperboloid()
{
    const float r0 = 100.f ;  // waist (z=0) radius 
    const float zf = 50.f ;   // at z=zf radius grows to  sqrt(2)*r0 
    const float z1 = -50.f ;  // z1 < z2 by assertion  
    const float z2 =  50.f ;  

    assert( z1 < z2 ); 

    Solid* solid = new Solid ; 
    solid->label = "hyperboloid" ; 
    solid->extent = r0*sqrt(2) ;  // guess  

    solid->prim = {} ; 
    solid->node = {} ;
    solid->node.q0.f = {r0, zf, z1, z2 } ; 
    solid->node.setTypecode(CSG_HYPERBOLOID) ; 

    return solid ; 
}

Solid* Solid::MakeBox3()
{
    float fx = 100.f ;   // fullside sizes
    float fy = 100.f ; 
    float fz = 150.f ; 

    Solid* solid = new Solid ; 
    solid->label = "box3" ; 
    solid->extent = 150.f ; 

    solid->prim = {} ; 
    solid->node = {} ;
    solid->node.q0.f = {fx, fy, fz, 0.f } ; 
    solid->node.setTypecode(CSG_BOX3) ; 

    return solid ; 
}

Solid* Solid::MakePlane()
{
    Solid* solid = new Solid ; 
    solid->label = "plane" ; 
    solid->extent = 150.f ;    // its unbounded 

    solid->prim = {} ; 
    solid->node = {} ; 
    solid->node.q0.f = {1.f, 0.f, 0.f, 0.f } ; // plane normal in +x direction, thru origin 
    solid->node.setTypecode(CSG_PLANE) ; 

    return solid ; 
}

Solid* Solid::MakeSlab()
{
    Solid* solid = new Solid ; 
    solid->label = "slab" ; 
    solid->extent = 150.f ;    // its unbounded 
    solid->prim = {} ; 
    solid->node = {} ; 

    solid->node.q0.f = {1.f, 0.f, 0.f, 0.f } ; // plane normal in +x direction
    solid->node.q1.f = {-10.f, 10.f, 0.f, 0.f } ; 
    solid->node.setTypecode(CSG_SLAB) ; 

    return solid ; 
}

Solid* Solid::MakeCylinder()
{
    Solid* solid = new Solid ; 
    solid->label = "cylinder" ; 

    float px = 0.f ; 
    float py = 0.f ; 
    float rxy = 100.f ; 

    float z1 = -50.f ; 
    float z2 =  50.f ; 

    solid->extent = 150.f ;   
    solid->prim = {} ; 
    solid->node = {} ; 

    solid->node.q0.f = { px, py, 0.f, rxy } ; 
    solid->node.q1.f = { z1, z2, 0.f, 0.f } ; 
    solid->node.setTypecode(CSG_CYLINDER); 

    return solid ; 
}

Solid* Solid::MakeDisc()
{
    Solid* solid = new Solid ; 
    solid->label = "disc" ; 

    float px = 0.f ; 
    float py = 0.f ; 

    float inner = 50.f ;
    float radius = 100.f ;  

    float z1 = -5.f ; 
    float z2 =  5.f ; 

    solid->extent = 100.f ;   
    solid->prim = {} ; 
    solid->node = {} ; 

    solid->node.q0.f = { px, py, inner, radius } ; 
    solid->node.q1.f = { z1, z2, 0.f, 0.f } ; 
    solid->node.setTypecode(CSG_DISC); 

    return solid ; 
}


void Solid::MakeSolids(std::vector<Solid*>& solids )
{
    solids.push_back( Solid::MakeSphere() ); 
    solids.push_back( Solid::MakeZSphere() ); 
    solids.push_back( Solid::MakeCone() ); 
    solids.push_back( Solid::MakeConvexPolyhedronCube() ); 
    solids.push_back( Solid::MakeConvexPolyhedronTetrahedron() ); 
    solids.push_back( Solid::MakeHyperboloid() ); 
    solids.push_back( Solid::MakeBox3() ); 
    solids.push_back( Solid::MakePlane() ); 
    solids.push_back( Solid::MakeSlab() ); 
    solids.push_back( Solid::MakeCylinder() ); 
    solids.push_back( Solid::MakeDisc() ); 
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

void test_intersect(std::vector<quad4>& recs, const Solid* solid, const float t_min, const float3& ray_origin, const float3& ray_direction )
{
    float4 isect = make_float4( 0.f, 0.f, 0.f, 0.f ) ; 
    bool valid_isect = intersect_node( isect, &solid->prim, &solid->node, solid->plan.data() , t_min, ray_origin, ray_direction ); 

    //dump(isect, valid_isect, ray_direction, ray_origin, solid->label.c_str() ); 

    quad4 rec ;  
    rec.q0.f = make_float4(ray_origin); 
    rec.q1.f = make_float4(ray_direction); 
    rec.q2.f = make_float4(0.f); 
    rec.q3.i = make_int4( valid_isect, 0, 0, 0);         

    if(valid_isect)
    {
        float t = isect.w ; 
        float3 pos = ray_origin + t*ray_direction ; 
        rec.q2.f = make_float4(pos, t ); 
    }

    recs.push_back(rec);  
}

void test_intersect(std::vector<quad4>& recs, const Solid* solid, const float t_min, const float3& ray_origin, const std::vector<float3>& dirs )
{
    for(unsigned i=0 ; i < dirs.size() ; i++)
    {
        const float3& ray_direction = dirs[i] ; 
        test_intersect( recs, solid, t_min, ray_origin, ray_direction ); 
    }
}

void axis_scan(const Solid* solid)
{
    std::vector<quad4> recs ; 

    float t_min = 0.f ;
    float3 origin = make_float3( 0.f, 0.f, 0.f ); 

    std::vector<float3> dirs ; 
    dirs.push_back( make_float3( 1.f, 0.f, 0.f));
    dirs.push_back( make_float3(-1.f, 0.f, 0.f));
    dirs.push_back( make_float3( 0.f, 1.f, 0.f));
    dirs.push_back( make_float3( 0.f,-1.f, 0.f));
    dirs.push_back( make_float3( 0.f, 0.f, 1.f));
    dirs.push_back( make_float3( 0.f, 0.f,-1.f));

    test_intersect(recs, solid,  t_min, origin, dirs );     

    std::string name = solid->label ; 
    name += ".npy" ; 

    std::cout << " recs.size " << recs.size() << std::endl ; 

    if(recs.size() > 0)
    {
        NP::Write("/tmp/intersect_node_tests/axis_scan", name.c_str(),  (float*)recs.data(), recs.size(), 4, 4 ); 
        recs.clear(); 
    }
}



void _rectangle_scan( std::vector<quad4>& recs, const Solid* solid, float t_min, unsigned n, float halfside, float y )
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
        test_intersect(recs,  solid,  t_min, z_top, z_down );     
        test_intersect(recs,  solid,  t_min, z_bot, z_up   );     

        x_lhs.z = v ; 
        x_rhs.z = v ; 
        test_intersect(recs,  solid,  t_min, x_lhs, x_right );     
        test_intersect(recs,  solid,  t_min, x_rhs, x_left  );     
    }
}


void rectangle_scan(const Solid* solid)
{
    std::vector<quad4> recs ; 
    float halfside = 2.0f*solid->extent ; 
    unsigned nxz = 100 ; 
    unsigned ny = 10 ; 
    float t_min = 0.f ;

    for(float y=-halfside ; y <= halfside ; y += halfside/float(ny) )
    {
        _rectangle_scan( recs, solid, t_min, nxz, halfside,   y );  
    }

    unsigned nhit = 0 ; 
    unsigned nmiss = 0 ; 

    for(unsigned i=0 ; i < recs.size() ; i++)
    {
        const quad4& rec = recs[i] ; 
        bool hit = rec.q3.i.x == 1  ; 
        if(hit)  nhit += 1 ; 
        if(!hit) nmiss += 1 ; 
    }
    std::cout 
        << " nhit " << nhit 
        << " nmiss " << nmiss 
        << std::endl 
        ;


    std::string name = solid->label ; 
    name += ".npy" ; 
    std::cout << " recs.size " << recs.size() << std::endl ; 
    if(recs.size() > 0)
    {
        NP::Write("/tmp/intersect_node_tests/rectangle_scan", name.c_str(),  (float*)recs.data(), recs.size(), 4, 4 ); 
        recs.clear(); 
    }
}







void circle_scan(const Solid* solid)
{
    std::vector<quad4> recs ; 

    float t_min = 0.f ;
    float radius = 2.0f*solid->extent ; 

    // M_PIf from sutil_vec_math.h
    for(float phi=0. ; phi <= M_PIf*2.0 ; phi+=M_PIf*2.0/1000.0 )
    {
        float3 origin = make_float3( radius*sin(phi), 0.f, radius*cos(phi) ); 
        float3 direction = make_float3( -sin(phi),  0.f, -cos(phi) ); 
        test_intersect(recs,  solid,  t_min, origin, direction );     
    }

    std::string name = solid->label ; 
    name += ".npy" ; 
    std::cout << " recs.size " << recs.size() << std::endl ; 

    if(recs.size() > 0)
    {
        NP::Write("/tmp/intersect_node_tests/circle_scan", name.c_str(),  (float*)recs.data(), recs.size(), 4, 4 ); 
        recs.clear(); 
    }
}


int main(int argc, char** argv)
{
    std::vector<Solid*> solids ; 
    if(argc > 1)
    {
         
        for(int s=1 ; s < argc ; s++)
        {
            char* arg = argv[s] ; 
            std::cout << " arg " << arg << std::endl ; 
            Solid* solid = Solid::Make(arg); 
            assert( solid ); 
            solids.push_back(solid); 
        }
    }
    else
    {
        Solid::MakeSolids(solids); 
    }

    unsigned ni = solids.size() ;
    for(unsigned i=0 ; i < ni ; i++)
    {
        Solid* solid = solids[i] ; 
        //axis_scan( solid ); 
        rectangle_scan( solid ); 
        //circle_scan( solid ); 
    }

    return 0 ;  
}
