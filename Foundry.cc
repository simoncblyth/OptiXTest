#include <iostream>
#include <iomanip>

#include "sutil_vec_math.h"
#include "OpticksCSG.h"
#include "Solid.h"
#include "CU.h"
#include "NP.hh"
#include "Foundry.h"


Foundry::Foundry()
    :
    d_node(nullptr),
    d_plan(nullptr),
    d_tran(nullptr),
    d_aabb(nullptr)
{
}

void Foundry::makeDemoSolids()
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
    dump(); 
}


void Foundry::write(const char* base, const char* rel) const 
{
    std::stringstream ss ;   
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   

    std::cout << "Foundry::write " << dir << std::endl ; 
    NP::Write(dir.c_str(), "solid.npy",    (int*)solid.data(), solid.size(), 4 ); 
    NP::Write(dir.c_str(), "prim.npy",     (int*)prim.data(),   prim.size(), 4 ); 
    NP::Write(dir.c_str(), "node.npy",   (float*)node.data(), node.size(), 4, 4 ); 
    NP::Write(dir.c_str(), "plan.npy",   (float*)plan.data(), plan.size(), 4 ); 
    NP::Write(dir.c_str(), "tran.npy",   (float*)tran.data(), tran.size(), 4, 4 ); 
    NP::Write(dir.c_str(), "aabb.npy",   (float*)aabb.data(), aabb.size(), 6 ); 
}

void Foundry::dump() const 
{
    for(unsigned idx=0 ; idx < getNumSolid() ; idx++) dumpSolid(idx); 
}

void Foundry::dumpSolid(int idx ) const 
{
    std::cout << "Foundry::dumpSolid " << idx << std::endl ; 
    const Solid* so = getSolid(idx); 
    std::cout << so->desc() << std::endl ; 

    for(unsigned primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const Prim* pr = getPrim(primIdx);   // numNode,nodeOffset,tranOffset,planOffset
        std::cout 
            << " primIdx " << std::setw(3) << primIdx << " "
            << pr->desc() 
            << "  "
            ; 

        for(unsigned nodeIdx=pr->nodeOffset ; nodeIdx < pr->nodeOffset+pr->numNode ; nodeIdx++)
        {
            const Node* nd = getNode(nodeIdx); 
            std::cout << nd->desc() << std::endl ; 
        }
    } 
}

/**
Collects aabb for all prim within a compound solid 

Although there is absolute addressing into solid, prim and node
that is only useful once you have the prim within the solid and node within
the prim.

**/

void Foundry::get_aabb( std::vector<float>& aabb, unsigned idx ) const 
{
    const Solid* so = getSolid(idx); 
    for(unsigned primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const Prim* pr = getPrim(primIdx);      // numNode,nodeOffset,tranOffset,planOffset
        unsigned nodeIdx0 = pr->nodeOffset ;    // first node for the prim, corresponds to root for node trees 
        const Node* nd = getNode(nodeIdx0); 
        assert(nd);
        const float* f = nd->AABB(); 
        for(int i=0 ; i < 6 ; i++ ) aabb.push_back(*(f+i)) ;  
    }
}


void Foundry::Dump( const std::vector<float>& aabb, const char* msg)  // static
{
    assert( aabb.size() % 6 == 0 ); 
    unsigned num_aabb = aabb.size() / 6 ;
    std::cout << "Foundry::Dump num_aabb " << num_aabb << " msg " << msg << std::endl ; 
    for(unsigned i=0 ; i < num_aabb ; i++)
    {   
        std::cout << std::setw(4) << i << " : " ; 
        for(unsigned j=0 ; j < 6 ; j++)  
            std::cout << std::setw(10) << std::fixed << std::setprecision(3) << *(aabb.data() + i*6 + j ) << " "  ;   
        std::cout << std::endl ; 
    }   
}

unsigned Foundry::getNumSolid() const { return solid.size(); } 
unsigned Foundry::getNumPrim() const  { return prim.size();  } 
unsigned Foundry::getNumNode() const  { return node.size(); }
unsigned Foundry::getNumPlan() const  { return plan.size(); }
unsigned Foundry::getNumTran() const  { return tran.size(); }

const Solid*  Foundry::getSolid(int idx_) const { 
    int idx = idx_ < 0 ? solid.size() + idx_ : idx_ ;   // -ve counts from end
    return idx < solid.size() ? solid.data() + idx : nullptr ; 
}   

const Prim*   Foundry::getPrim(unsigned primIdx)   const { return primIdx  < prim.size()  ? prim.data()  + primIdx  : nullptr ; } 
const Node*   Foundry::getNode(unsigned nodeIdx)   const { return nodeIdx  < node.size()  ? node.data()  + nodeIdx  : nullptr ; }  
const float4* Foundry::getPlan(unsigned planIdx)   const { return planIdx  < plan.size()  ? plan.data()  + planIdx  : nullptr ; }
const qat4*   Foundry::getTran(unsigned tranIdx)   const { return tranIdx  < tran.size()  ? tran.data()  + tranIdx  : nullptr ; }

const Solid* Foundry::getSolid(const char* name) const  // caution stored labels truncated to 4 char 
{
    unsigned missing = ~0u ; 
    unsigned idx = missing ; 
    for(unsigned i=0 ; i < solid.size() ; i++) if(strcmp(solid[i].label, name) == 0) idx = i ;  
    assert( idx != missing ); 
    return getSolid(idx) ; 
}

unsigned Foundry::make(char type)
{
    unsigned idx = ~0u ; 
    switch(type)
    {
       case 'S':  idx = makeSphere()         ; break ;    
       case 'Z':  idx = makeZSphere()        ; break ;    
       case 'O':  idx = makeCone()           ; break ;    
       case 'H':  idx = makeHyperboloid()    ; break ;    
       case 'B':  idx = makeBox3()           ; break ;    
       case 'P':  idx = makePlane()          ; break ;    
       case 'A':  idx = makeSlab()           ; break ;    
       case 'Y':  idx = makeCylinder()       ; break ;    
       case 'D':  idx = makeDisc()                        ; break ;    
       case 'U':  idx = makeConvexPolyhedronCube()        ; break ;    
       case 'T':  idx = makeConvexPolyhedronTetrahedron() ; break ;    
    }
    assert( idx != ~0u ); 
    return idx ; 
}

unsigned Foundry::make(const char* name)
{
    if(     strcmp(name, "sphere") == 0)           return makeSphere(name) ;
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

**/

void Foundry::addNode(Node& nd, const std::vector<float4>* pl )
{
    unsigned num_planes = pl ? pl->size() : 0 ; 
    if(num_planes > 0)
    {
        nd.setPlaneNum(num_planes);    
        nd.setTypecode(CSG_CONVEXPOLYHEDRON) ; 
        for(unsigned i=0 ; i < num_planes ; i++) plan.push_back((*pl)[i]);  
    }

    node.push_back(nd); 
}

void Foundry::addNodes(const std::vector<Node>& nds )
{
    for(unsigned i=0 ; i < nds.size() ; i++) 
    {
        const Node& nd = nds[i]; 
        node.push_back(nd); 
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


/**
Foundary::makeLayered
----------------------------

Once have transforms working can generalize to any shape. 
But prior to that just do layering for sphere for adiabatic transition
from Shape to Foundry/Solid.

NB Each layer is a separate Prim with a single Node 

**/

unsigned Foundry::makeLayered(const char* label, float outer_radius, unsigned layers )
{
    std::vector<float> radii ;
    for(unsigned i=0 ; i < layers ; i++) radii.push_back(outer_radius*float(layers-i)/float(layers)) ; 

    Solid so(label, layers, prim.size(), outer_radius ) ; 

    for(unsigned i=0 ; i < layers ; i++)
    {
        addPrim(1); 
        float radius = radii[i]; 

        if(strcmp(label, "sphere") == 0)
        {
            Node nd = Node::Sphere(radius); 
            addNode(nd); 
        }
        else if(strcmp(label, "zsphere") == 0)
        {
            Node nd = Node::ZSphere(radius, -radius/2.f , radius/2.f ); 
            addNode(nd); 
        }
        else
        {
            assert( 0 && "layered only implemented for sphere and zsphere currently" ); 
        } 
    }

    unsigned idx = solid.size(); 
    solid.push_back(so); 
    return idx ; 
}


/**
Foundry::makeSolid11 makes 1-Prim with 1-Node
------------------------------------------------
**/

unsigned Foundry::makeSolid11(float extent, const char* label, Node& nd, const std::vector<float4>* pl  ) 
{
    Solid so(label, 1, prim.size(), extent) ; 

    unsigned num_node = 1 ; 
    addPrim(num_node); 
    addNode(nd, pl ); 

    unsigned idx = solid.size(); 
    solid.push_back(so); 
    return idx ; 
}

unsigned Foundry::makeSphere(const char* label, float radius)
{
    Node nd = Node::Sphere(radius); 
    float extent = radius ;  // TODO: get from AABB
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeZSphere(const char* label, float radius, float z1, float z2)
{
    Node nd = Node::ZSphere(radius, z1, z2); 
    float extent = radius ;  // TODO: get from AABB
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeCone(const char* label, float r1, float z1, float r2, float z2)
{
    float extent = 500.f ;  // guess 
    Node nd = Node::Cone(r1, z1, r2, z2 ); 
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeHyperboloid(const char* label, float r0, float zf, float z1, float z2)
{
    float extent = r0*sqrt(2.) ;  // guess  TODO: get from AABB
    Node nd = Node::Hyperboloid( r0, zf, z1, z2 ); 
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeBox3(const char* label, float fx, float fy, float fz )
{
    float extent = 150.f ; 
    Node nd = Node::Box3(fx, fy, fz); 
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makePlane(const char* label, float nx, float ny, float nz, float d)
{
    float extent = 150.f ;    // its unbounded 
    Node nd = Node::Plane(nx, ny, nz, d ); 
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeSlab(const char* label, float nx, float ny, float nz, float d1, float d2 )
{
    float extent = 150.f ;   // hmm: its unbounded 
    Node nd = Node::Slab( nx, ny, nz, d1, d1 ); 
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeCylinder(const char* label, float px, float py, float radius, float z1, float z2)
{
    float extent = 100.f ; 
    Node nd = Node::Cylinder( px, py, radius, z1, z2 ); 
    return makeSolid11(extent, label, nd, nullptr ); 
}

unsigned Foundry::makeDisc(const char* label, float px, float py, float ir, float r, float z1, float z2)
{
    Node nd = Node::Disc(px, py, ir, r, z1, z2 ); 
    float extent = 100.f ; 
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

          Faces (right-hand-rule oriented outwards normals)
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


void Foundry::upload()
{
    unsigned num_node = node.size(); 
    unsigned num_plan = plan.size(); 
    unsigned num_tran = tran.size(); 

    std::cout 
        << "Foundry::upload"
        << " num_node " << num_node
        << " num_plan " << num_plan
        << " num_tran " << num_tran
        << std::endl
        ;

    d_node = num_node > 0 ? CU::UploadArray<Node>(node.data(), num_node ) : nullptr ; 
    d_plan = num_plan > 0 ? CU::UploadArray<float4>(plan.data(), num_plan ) : nullptr ; 
    d_tran = num_tran > 0 ? CU::UploadArray<qat4>(tran.data(), num_tran ) : nullptr ; 
}


