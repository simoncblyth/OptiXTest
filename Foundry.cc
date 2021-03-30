#include <iostream>
#include <iomanip>

#include "sutil_vec_math.h"
#include "OpticksCSG.h"
#include "Solid.h"
#include "Foundry.h"


Foundry::Foundry()
{
   // not calling init as in real usage need to minimize contained nodes, prim, tran etc..
}


void Foundry::init()
{
    makeSolids(); 
    dump(); 
}

void Foundry::makeSolids()
{
    makeSphere(); 
    makeZSphere(); 
    makeCone(); 
    makeHyperboloid(); 
    makeBox3(); 
    makePlane(); 
    makeSlab(); 
    makeCylinder() ; 
    makeDisc(); 
    makeConvexPolyhedronCube(); 
    makeConvexPolyhedronTetrahedron(); 
}

void Foundry::dump() const 
{
    for(unsigned idx=0 ; idx < getNumSolid() ; idx++)
    {
        const Solid* so = getSolid(idx); 
        std::cout 
            << std::setw(3) << idx 
            << " : "
            << so->desc()
            << std::endl
            ;
    }
}

unsigned Foundry::getNumSolid() const { return solid.size(); }
unsigned Foundry::getNumPrim() const  { return prim.size();  }
unsigned Foundry::getNumNode() const  { return node.size(); }
unsigned Foundry::getNumPlan() const  { return plan.size(); }
unsigned Foundry::getNumTran() const  { return tran.size(); }

const Solid*  Foundry::getSolid(unsigned solidIdx) const { return solidIdx < solid.size() ? solid.data() + solidIdx : nullptr ; }   
const Prim*   Foundry::getPrim(unsigned primIdx)   const { return primIdx  < prim.size()  ? prim.data()  + primIdx  : nullptr ; } 
const Node*   Foundry::getNode(unsigned nodeIdx)   const { return nodeIdx  < node.size()  ? node.data()  + nodeIdx  : nullptr ; }  
const float4* Foundry::getPlan(unsigned planIdx)   const { return planIdx  < plan.size()  ? plan.data()  + planIdx  : nullptr ; }
const quad4*  Foundry::getTran(unsigned tranIdx)   const { return tranIdx  < tran.size()  ? tran.data()  + tranIdx  : nullptr ; }

const Solid* Foundry::getSolid(const char* name) const 
{
    unsigned missing = ~0u ; 
    unsigned idx = missing ; 
    for(unsigned i=0 ; i < solid.size() ; i++) if(strcmp(solid[i].label, name) == 0) idx = i ;  
    assert( idx != missing ); 
    return getSolid(idx) ; 
}


unsigned Foundry::make(const char* name)
{
    if(     strcmp(name, "sphere") == 0)           return makeSphere(name, 100.f) ;
    else if(strcmp(name, "zsphere") == 0)          return makeZSphere(name) ;
    else if(strcmp(name, "cone") == 0)             return makeCone(name) ;
    else if(strcmp(name, "hyperboloid") == 0)      return makeHyperboloid(name) ;
    else if(strcmp(name, "box3") == 0)             return makeBox3(name) ;
    else if(strcmp(name, "plane") == 0)            return makePlane(name) ;
    else if(strcmp(name, "slab") == 0)             return makeSlab(name) ;
    else if(strcmp(name, "cylinder") == 0)         return makeCylinder(name) ;
    else if(strcmp(name, "disc") == 0)             return makeDisc(name) ;
    else if(strcmp(name, "convexpolyhedron_cube") == 0) return makeConvexPolyhedronCube(name) ;
    else if(strcmp(name, "convexpolyhedron_tetrahedron") == 0) return makeConvexPolyhedronTetrahedron(name) ;
    else assert(0) ;
    return ~0u ;  
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


unsigned Foundry::makeSolid11(float extent, const char* label, Node& nd, const std::vector<float4>* pl  ) 
{
    Solid sol = {} ; 
    sol.label = strdup(label) ; 
    sol.numPrim = 1 ; 
    sol.primOffset = prim.size(); 
    sol.extent = extent ;

    unsigned num_node = 1 ; 
    addPrim(num_node); 
    addNode(nd, true, pl ); 

    unsigned idx = solid.size(); 
    solid.push_back(sol); 
    return idx ; 
}

unsigned Foundry::makeSphere(const char* label, float radius)
{
    assert( radius > 0.f); 
    Node nd = {} ;
    nd.setParam( 0.f, 0.f, 0.f, radius,  0.f,  0.f ); 
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  ); 
    nd.setTypecode(CSG_SPHERE) ; 
    float extent = radius ;  // TODO: get from AABB
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeZSphere(const char* label, float radius, float z1, float z2)
{
    assert( radius > 0.f); 
    assert( z2 > z1 ); 
    Node nd = {} ;
    nd.setParam( 0.f, 0.f, 0.f, radius, z1, z2 ); 
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  ); 
    nd.setTypecode(CSG_ZSPHERE) ; 

    float extent = 100.f ; 
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeCone(const char* label, float r1, float z1, float r2, float z2)
{
    float extent = 500.f ;  // guess 
    assert( z2 > z1 ); 
    float rmax = fmaxf(r1, r2) ;
    Node nd = {} ;
    nd.setParam( r1, z1, r2, z2, 0.f, 0.f ) ;
    nd.setAABB( -rmax, -rmax, z1, rmax, rmax, z2 ); 
    nd.setTypecode(CSG_CONE) ; 

    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeHyperboloid(const char* label, float r0, float zf, float z1, float z2)
{
    assert( z1 < z2 ); 

    const float rr0 = r0*r0 ; 
    const float z1s = z1/zf ; 
    const float z2s = z2/zf ; 

    const float rr1 = rr0 * ( z1s*z1s + 1.f ) ;
    const float rr2 = rr0 * ( z2s*z2s + 1.f ) ;
    const float rmx = sqrtf(fmaxf( rr1, rr2 )) ; 

    float extent = r0*sqrt(2.) ;  // guess  TODO: get from AABB

    Node nd = {} ;
    nd.setParam(r0, zf, z1, z2, 0.f, 0.f ) ; 
    nd.setAABB(  -rmx,  -rmx,  z1,  rmx, rmx, z2 ); 
    nd.setTypecode(CSG_HYPERBOLOID) ; 

    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeBox3(const char* label, float fx, float fy, float fz )
{
    float extent = 150.f ; 
    assert( fx > 0.f ); 
    assert( fy > 0.f ); 
    assert( fz > 0.f ); 

    Node nd = {} ;
    nd.setParam( fx, fy, fz, 0.f, 0.f, 0.f ); 
    nd.setAABB( -fx*0.5f , -fy*0.5f, -fz*0.5f, fx*0.5f , fy*0.5f, fz*0.5f );   
    nd.setTypecode(CSG_BOX3) ; 

    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makePlane(const char* label, float nx, float ny, float nz, float d)
{
    float extent = 150.f ;    // its unbounded 

    Node nd = {} ;
    nd.setParam(nx, ny, nz, d, 0.f, 0.f ) ;
    nd.setTypecode(CSG_PLANE) ; 

    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeSlab(const char* label, float nx, float ny, float nz, float d1, float d2 )
{
    float extent = 150.f ;   // hmm: its unbounded 
    Node nd = {} ;
    nd.setParam( nx, ny, nz, 0.f, d1, d2 ); 

    nd.setTypecode(CSG_SLAB) ; 

    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeCylinder(const char* label, float px, float py, float radius, float z1, float z2)
{
    float extent = 100.f ; 

    Node nd = {} ; 
    nd.setParam( px, py, 0.f, radius, z1, z2)  ; 
    nd.setTypecode(CSG_CYLINDER); 

    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeDisc(const char* label, float px, float py, float ir, float r, float z1, float z2)
{
    float extent = 100.f ; 
    Node nd = {} ;
    nd.setParam( px, py, ir, r, z1, z2 ); 
    nd.setAABB( px - r , py - r , z1, px + r, py + r, z2 ); 
    nd.setTypecode(CSG_DISC); 

    return makeSolid11(extent, label, nd, nullptr ); 
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
    //std::cout << " di " << di << " dj " << dj << " dk " << dk << " n (" << n.x << "," << n.y << "," << n.z << ")" << std::endl ; 
    float4 plane = make_float4( n, di ) ; 
    return plane ;  
}

unsigned Foundry::makeConvexPolyhedronCube(const char* label)
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
    return makeSolid11(extent, label, nd, &pl ); 
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


unsigned Foundry::makeConvexPolyhedronTetrahedron(const char* label)
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

    //for(unsigned i=0 ; i < pl.size() ; i++) std::cout << " pl (" << pl[i].x << "," << pl[i].y << "," << pl[i].z << "," << pl[i].w << ") " << std::endl ;
 
    Node nd = {} ;
    return makeSolid11(extent, label, nd, &pl ); 
}


