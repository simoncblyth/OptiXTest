#include <iostream>
#include "sutil_vec_math.h"
#include "OpticksCSG.h"
#include "Solid.h"
#include "Foundry.h"

void Foundry::makeSolids()
{
    solid.push_back( *makeSphere(100.f) ); 
    solid.push_back( *makeZSphere() ); 
    solid.push_back( *makeCone() ); 
    solid.push_back( *makeHyperboloid() ); 
    solid.push_back( *makeBox3() ); 
    solid.push_back( *makePlane() ); 
    solid.push_back( *makeSlab() ); 
    solid.push_back( *makeCylinder() ); 
    solid.push_back( *makeDisc() ); 
    solid.push_back( *makeConvexPolyhedronCube() ); 
    solid.push_back( *makeConvexPolyhedronTetrahedron() ); 
}

Solid* Foundry::make(const char* name)
{
    if(     strcmp(name, "sphere") == 0)           return makeSphere(100.f) ;
    else if(strcmp(name, "zsphere") == 0)          return makeZSphere() ;
    else if(strcmp(name, "cone") == 0)             return makeCone() ;
    else if(strcmp(name, "hyperboloid") == 0)      return makeHyperboloid() ;
    else if(strcmp(name, "box3") == 0)             return makeBox3() ;
    else if(strcmp(name, "plane") == 0)            return makePlane() ;
    else if(strcmp(name, "slab") == 0)             return makeSlab() ;
    else if(strcmp(name, "cylinder") == 0)         return makeCylinder() ;
    else if(strcmp(name, "disc") == 0)             return makeDisc() ;
    else if(strcmp(name, "convexpolyhedron_cube") == 0) return makeConvexPolyhedronCube() ;
    else if(strcmp(name, "convexpolyhedron_tetrahedron") == 0) return makeConvexPolyhedronTetrahedron() ;
    else return nullptr ; 
}


/**
Foundry::addNode
--------------------

This is called after addPrim allowing an offset check to be made.

**/

void Foundry::addNode(Node& nd, bool prim_check, const std::vector<float4>* pl )
{
    unsigned num_planes = pl ? pl->size() : 0 ; 
    if(num_planes > 0)
    {
        nd.setPlaneNum(num_planes);    
        nd.setTypecode(CSG_CONVEXPOLYHEDRON) ; 
        for(unsigned i=0 ; i < num_planes ; i++) plan.push_back((*pl)[i]);  
    }

    node.push_back(nd); 

    if(prim_check)
    {
       const Prim& pr = prim.back(); 
       assert( node.size() - pr.nodeOffset == 1 ); 
       assert( plan.size() - pr.planOffset == num_planes ); 
    }
}

void Foundry::addNodes(const std::vector<Node>& nds, bool prim_check )
{
    for(unsigned i=0 ; i < nds.size() ; i++) 
    {
        const Node& nd = nds[i]; 
        node.push_back(nd); 
    }

    if(prim_check)
    {
       const Prim& pr = prim.back(); 
       assert( node.size() - pr.nodeOffset == nds.size() ); 
    }
}

/**
Foundry::addPrim
------------------

Offsets counts for  node, tran and plan are 
persisted into the Prim. 
Thus must addPrim prior to adding any node, 
tran or plan needed for a prim.

**/

void Foundry::addPrim(int num_node)  
{
    Prim pr = {} ;
    pr.numNode = num_node ; 
    pr.nodeOffset = node.size(); 
    pr.tranOffset = tran.size(); 
    pr.planOffset = plan.size(); 
    prim.push_back(pr); 
}


Solid* Foundry::makeSolid11(float extent, const char* label, Node& nd, const std::vector<float4>* pl  ) 
{
    Solid* solid = new Solid ; 
    solid->label = strdup(label) ; 
    solid->numPrim = 1 ; 
    solid->primOffset = prim.size(); 
    solid->extent = extent ;
    solid->foundry = this ; 

    unsigned num_node = 1 ; 
    addPrim(num_node); 
    addNode(nd, true, pl ); 

    return solid ; 
}

Solid* Foundry::makeSphere(float radius)
{
    Node nd = {} ;
    nd.q0.f = {0.f,   0.f, 0.f, radius } ; 
    nd.setTypecode(CSG_SPHERE) ; 
    float extent = radius ; 
    return makeSolid11(extent, "sphere", nd, nullptr ); 
}

Solid* Foundry::makeZSphere()
{
    Node nd = {} ;
    nd.q0.f = {0.f,   0.f, 0.f, 100.f } ; 
    nd.q1.f = {-50.f,50.f, 0.f,   0.f } ; 
    nd.setTypecode(CSG_ZSPHERE) ; 
    float extent = 100.f ; 
    return makeSolid11(extent, "zsphere", nd, nullptr ); 
}

Solid* Foundry::makeCone()
{
    float extent = 500.f ;  // guess 

    float r2 = 100.f ;
    float r1 = 300.f ;
    float z2 = -100.f ;  
    float z1 = -300.f ;
    assert( z2 > z1 ); 
    //float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex

    Node nd = {} ;
    nd.q0.f = {r1, z1, r2, z2 } ; 
    nd.setTypecode(CSG_CONE) ; 

    return makeSolid11(extent, "cone", nd, nullptr ); 
}

Solid* Foundry::makeHyperboloid()
{
    const float r0 = 100.f ;  // waist (z=0) radius 
    const float zf = 50.f ;   // at z=zf radius grows to  sqrt(2)*r0 
    const float z1 = -50.f ;  // z1 < z2 by assertion  
    const float z2 =  50.f ; 
    assert( z1 < z2 ); 
 
    float extent = r0*sqrt(2.) ;  // guess  

    Node nd = {} ;
    nd.q0.f = {r0, zf, z1, z2 } ; 
    nd.setTypecode(CSG_HYPERBOLOID) ; 

    return makeSolid11(extent, "hyperboloid", nd, nullptr ); 
}

Solid* Foundry::makeBox3()
{
    float extent = 150.f ; 

    float fx = 100.f ;   // fullside sizes
    float fy = 100.f ; 
    float fz = 150.f ; 

    Node nd = {} ;
    nd.q0.f = {fx, fy, fz, 0.f } ; 
    nd.setTypecode(CSG_BOX3) ; 

    return makeSolid11(extent, "box3", nd, nullptr ); 
}

Solid* Foundry::makePlane()
{
    float extent = 150.f ;    // its unbounded 

    Node nd = {} ;
    nd.q0.f = {1.f, 0.f, 0.f, 0.f } ; // plane normal in +x direction, thru origin 
    nd.setTypecode(CSG_PLANE) ; 

    return makeSolid11(extent, "plane", nd, nullptr ); 
}

Solid* Foundry::makeSlab()
{
    float extent = 150.f ;   // hmm: its unbounded 
    Node nd = {} ;
    nd.q0.f = {1.f, 0.f, 0.f, 0.f } ; // plane normal in +x direction
    nd.q1.f = {-10.f, 10.f, 0.f, 0.f } ; 
    nd.setTypecode(CSG_SLAB) ; 

    return makeSolid11(extent, "slab", nd, nullptr ); 
}

Solid* Foundry::makeCylinder()
{
    float extent = 100.f ; 

    float px = 0.f ; 
    float py = 0.f ; 
    float rxy = 100.f ; 
    float z1 = -50.f ; 
    float z2 =  50.f ; 

    Node nd = {} ; 
    nd.q0.f = { px, py, 0.f, rxy } ; 
    nd.q1.f = { z1, z2, 0.f, 0.f } ; 
    nd.setTypecode(CSG_CYLINDER); 

    return makeSolid11(extent, "cylinder", nd, nullptr ); 
}

Solid* Foundry::makeDisc()
{
    float extent = 100.f ; 

    float px = 0.f ; 
    float py = 0.f ; 
    float inner = 50.f ;
    float radius = 100.f ;  
    float z1 = -5.f ; 
    float z2 =  5.f ; 

    Node nd = {} ;
    nd.q0.f = { px, py, inner, radius } ; 
    nd.q1.f = { z1, z2, 0.f, 0.f } ; 
    nd.setTypecode(CSG_DISC); 

    return makeSolid11(extent, "disc", nd, nullptr ); 
}





float4 Foundry::TriPlane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k )  // static 
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


Solid* Foundry::makeConvexPolyhedronCube()
{
    float hx = 10.f ; 
    float hy = 20.f ; 
    float hz = 30.f ; 

    std::vector<float4> pl ; 
    pl.push_back( make_float4(  1.f,  0.f,  0.f, hx ) ); 
    pl.push_back( make_float4( -1.f,  0.f,  0.f, hx ) ); 
    pl.push_back( make_float4(  0.f,  1.f,  0.f, hy ) ); 
    pl.push_back( make_float4(  0.f, -1.f,  0.f, hy ) ); 
    pl.push_back( make_float4(  0.f,  0.f,  1.f, hz ) ); 
    pl.push_back( make_float4(  0.f,  0.f, -1.f, hz ) );

    float extent = 30.f ;  
    Node nd = {} ;
    return makeSolid11(extent, "convexpolyhedron_cube", nd, &pl ); 
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

*/


Solid* Foundry::makeConvexPolyhedronTetrahedron()
{
    float s = 100.f*sqrt(3) ; 
    float extent = s ; 

    std::vector<float3> vtx ; 
    vtx.push_back(make_float3( s, s, s));  
    vtx.push_back(make_float3( s,-s,-s)); 
    vtx.push_back(make_float3(-s, s,-s)); 
    vtx.push_back(make_float3(-s,-s, s)); 

    std::vector<float4> pl ; 
    pl.push_back(TriPlane(vtx, 0, 1, 2)) ;  
    pl.push_back(TriPlane(vtx, 1, 3, 2)) ;  
    pl.push_back(TriPlane(vtx, 3, 0, 2)) ;  
    pl.push_back(TriPlane(vtx, 0, 3, 1)) ;  

    for(unsigned i=0 ; i < pl.size() ; i++) 
    {
        const float4& p = pl[i] ;  
        std::cout << " pl (" << p.x << "," << p.y << "," << p.z << "," << p.w << ") " << std::endl ;
    } 

    Node nd = {} ;
    return makeSolid11(extent, "convexpolyhedron_tetrahedron", nd, &pl ); 
}


