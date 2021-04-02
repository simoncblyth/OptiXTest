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
    imax(1000),
    d_solid(nullptr),
    d_prim(nullptr),
    d_node(nullptr),
    d_plan(nullptr),
    d_tran(nullptr)
{
    init(); 
}

void Foundry::init()
{
    // without sufficient reserved the vectors may reallocate on any push_back invalidating prior pointers 
    solid.reserve(imax); 
    prim.reserve(imax); 
    node.reserve(imax); 
    plan.reserve(imax); 
    tran.reserve(imax); 
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

void Foundry::dump() const 
{
    for(unsigned idx=0 ; idx < solid.size() ; idx++) dumpSolid(idx); 
}

void Foundry::dumpSolid(unsigned solidIdx) const 
{
    std::cout << "Foundry::dumpSolid " << solidIdx << std::endl ; 

    const Solid* so = solid.data() + solidIdx ; 
    std::cout << so->desc() << std::endl ; 

    for(unsigned primIdx=so->primOffset ; primIdx < so->primOffset+so->numPrim ; primIdx++)
    {
        const Prim* pr = prim.data() + primIdx ; 
        std::cout 
            << " primIdx " << std::setw(3) << primIdx << " "
            << pr->desc() 
            << std::endl 
            ; 

        for(unsigned nodeIdx=pr->nodeOffset() ; nodeIdx < pr->nodeOffset()+pr->numNode() ; nodeIdx++)
        {
            const Node* nd = node.data() + nodeIdx ; 
            std::cout << nd->desc() << std::endl ; 
        }
    } 
}






/**
Foundry::getPrimSpec
----------------------

Provides the specification to access the AABB and sbtIndexOffset of all Prim 
of a Solid.  The specification includes pointers, counts and stride.

NB PrimAABB is distinct from NodeAABB. Cannot directly use NodeAABB 
because the number of nodes for each prim (node tree) varies meaning 
that the strides are irregular. 

**/

PrimSpec Foundry::getPrimSpec(unsigned solidIdx) const 
{
    PrimSpec ps = d_prim ? getPrimSpecDevice(solidIdx) : getPrimSpecHost(solidIdx) ; 
    if(ps.device == false) std::cout << "Foundry::getPrimSpec WARNING using host PrimSpec " << std::endl ; 
    return ps ; 
}
PrimSpec Foundry::getPrimSpecHost(unsigned solidIdx) const 
{
    const Solid* so = solid.data() + solidIdx ; 
    PrimSpec ps = Prim::MakeSpec( prim.data(),  so->primOffset, so->numPrim ); ; 
    ps.device = false ; 
    return ps ; 
}
PrimSpec Foundry::getPrimSpecDevice(unsigned solidIdx) const 
{
    assert( d_prim ); 
    const Solid* so = solid.data() + solidIdx ; 
    PrimSpec ps = Prim::MakeSpec( d_prim,  so->primOffset, so->numPrim ); ; 
    ps.device = true ; 
    return ps ; 
}


unsigned Foundry::getNumSolid() const { return solid.size(); } 
unsigned Foundry::getNumPrim() const  { return prim.size();  } 
unsigned Foundry::getNumNode() const  { return node.size(); }
unsigned Foundry::getNumPlan() const  { return plan.size(); }
unsigned Foundry::getNumTran() const  { return tran.size(); }

const Solid*  Foundry::getSolid(unsigned solidIdx) const { return solidIdx < solid.size() ? solid.data() + solidIdx  : nullptr ; } 
const Prim*   Foundry::getPrim(unsigned primIdx)   const { return primIdx  < prim.size()  ? prim.data()  + primIdx  : nullptr ; } 
const Node*   Foundry::getNode(unsigned nodeIdx)   const { return nodeIdx  < node.size()  ? node.data()  + nodeIdx  : nullptr ; }  
const float4* Foundry::getPlan(unsigned planIdx)   const { return planIdx  < plan.size()  ? plan.data()  + planIdx  : nullptr ; }
const qat4*   Foundry::getTran(unsigned tranIdx)   const { return tranIdx  < tran.size()  ? tran.data()  + tranIdx  : nullptr ; }


const Solid*  Foundry::getSolid_(int solidIdx_) const { 
    unsigned solidIdx = solidIdx_ < 0 ? unsigned(solid.size() + solidIdx_) : unsigned(solidIdx_)  ;   // -ve counts from end
    return getSolid(solidIdx); 
}   

const Solid* Foundry::getSolid(const char* name) const  // caution stored labels truncated to 4 char 
{
    unsigned missing = ~0u ; 
    unsigned idx = missing ; 
    for(unsigned i=0 ; i < solid.size() ; i++) if(strcmp(solid[i].label, name) == 0) idx = i ;  
    assert( idx != missing ); 
    return getSolid(idx) ; 
}

/**
Foundry::getSolidIdx
----------------------

Without sufficient reserve allocation this is unreliable as pointers go stale on reallocations.

**/

unsigned Foundry::getSolidIdx(const Solid* so) const 
{
    unsigned idx = ~0u ; 
    for(unsigned i=0 ; i < solid.size() ; i++) 
    {
       const Solid* s = solid.data() + i ; 
       std::cout << " i " << i << " s " << s << " so " << so << std::endl ; 
       if(s == so) idx = i ;  
    } 
    assert( idx != ~0u ); 
    return idx ; 
}








Solid* Foundry::make(char type)
{
    Solid* so = nullptr ; 
    switch(type)
    {
       case 'S':  so = makeSphere()         ; break ;    
       case 'Z':  so = makeZSphere()        ; break ;    
       case 'O':  so = makeCone()           ; break ;    
       case 'H':  so = makeHyperboloid()    ; break ;    
       case 'B':  so = makeBox3()           ; break ;    
       case 'P':  so = makePlane()          ; break ;    
       case 'A':  so = makeSlab()           ; break ;    
       case 'Y':  so = makeCylinder()       ; break ;    
       case 'D':  so = makeDisc()                        ; break ;    
       case 'U':  so = makeConvexPolyhedronCube()        ; break ;    
       case 'T':  so = makeConvexPolyhedronTetrahedron() ; break ;    
    }
    assert( so  ); 
    return so ; 
}

Solid* Foundry::make(const char* name)
{
    Solid* so = nullptr ; 
    if(     strcmp(name, "sphere") == 0)           so = makeSphere(name) ;
    else if(strcmp(name, "zsphere") == 0)          so = makeZSphere(name) ;
    else if(strcmp(name, "cone") == 0)             so = makeCone(name) ;
    else if(strcmp(name, "hyperboloid") == 0)      so = makeHyperboloid(name) ;
    else if(strcmp(name, "box3") == 0)             so = makeBox3(name) ;
    else if(strcmp(name, "plane") == 0)            so = makePlane(name) ;
    else if(strcmp(name, "slab") == 0)             so = makeSlab(name) ;
    else if(strcmp(name, "cylinder") == 0)         so = makeCylinder(name) ;
    else if(strcmp(name, "disc") == 0)             so = makeDisc(name) ;
    else if(strcmp(name, "convexpolyhedron_cube") == 0) so = makeConvexPolyhedronCube(name) ;
    else if(strcmp(name, "convexpolyhedron_tetrahedron") == 0) so = makeConvexPolyhedronTetrahedron(name) ;
    assert( so ); 
    return so ;  
}


/**
Foundry::addNode
--------------------

**/

Node* Foundry::addNode(Node nd, const std::vector<float4>* pl )
{
    unsigned num_planes = pl ? pl->size() : 0 ; 
    if(num_planes > 0)
    {
        nd.setPlaneNum(num_planes);    
        nd.setTypecode(CSG_CONVEXPOLYHEDRON) ; 
        for(unsigned i=0 ; i < num_planes ; i++) addPlan((*pl)[i]);  
    }

    unsigned idx = node.size() ;  
    assert( idx < imax ); 
    node.push_back(nd); 
    return node.data() + idx ; 
}

Node* Foundry::addNodes(const std::vector<Node>& nds )
{
    unsigned idx = node.size() ; 
    for(unsigned i=0 ; i < nds.size() ; i++) 
    {
        const Node& nd = nds[i]; 
        idx = node.size() ;  
        assert( idx < imax ); 
        node.push_back(nd); 
    }
    return node.data() + idx ; 
}

/**
Foundry::addPrim
------------------

Offsets counts for  node, tran and plan are 
persisted into the Prim. 
Thus must addPrim prior to adding any node, 
tran or plan needed for a prim.

**/

Prim* Foundry::addPrim(int num_node)  
{
    Prim pr = {} ;
    pr.setNumNode(num_node) ; 
    pr.setNodeOffset(node.size()); 
    pr.setTranOffset(tran.size()); 
    pr.setPlanOffset(plan.size()); 
    pr.setSbtIndexOffset(0) ; 

    unsigned primIdx = prim.size(); 
    assert( primIdx < imax ); 
    prim.push_back(pr); 
    return prim.data() + primIdx ; 
}

Solid* Foundry::addSolid(unsigned num_prim, const char* label )
{
    unsigned idx = solid.size(); 
    assert( idx < imax ); 

    unsigned primOffset = prim.size(); 
    Solid so = {} ; 
    memcpy( so.label, label, 4 ); 
    so.numPrim = num_prim ; 
    so.primOffset = primOffset ; 
    so.extent = 0.f ; 

    solid.push_back(so); 
    return solid.data() + idx  ; 
}

float4* Foundry::addPlan(const float4& pl )
{
    unsigned idx = plan.size(); 
    assert( idx < imax ); 
    plan.push_back(pl); 
    return plan.data() + idx ; 
}



/**
Foundary::makeLayered
----------------------------

Once have transforms working can generalize to any shape. 
But prior to that just do layering for sphere for adiabatic transition
from Shape to Foundry/Solid.

NB Each layer is a separate Prim with a single Node 

**/

Solid* Foundry::makeLayered(const char* label, float outer_radius, unsigned layers )
{
    std::vector<float> radii ;
    for(unsigned i=0 ; i < layers ; i++) radii.push_back(outer_radius*float(layers-i)/float(layers)) ; 

    unsigned numPrim = layers ; 
    Solid* so = addSolid(numPrim, label); 
    so->extent = outer_radius ; 

    for(unsigned i=0 ; i < numPrim ; i++)
    {
        unsigned numNode = 1 ; 
        Prim* p = addPrim(numNode); 
        float radius = radii[i]; 

        Node* n = nullptr ; 

        if(strcmp(label, "sphere") == 0)
        {
            n = addNode(Node::Sphere(radius)); 
        }
        else if(strcmp(label, "zsphere") == 0)
        {
            n = addNode(Node::ZSphere(radius, -radius/2.f , radius/2.f )); 
        }
        else
        {
            assert( 0 && "layered only implemented for sphere and zsphere currently" ); 
        } 

        p->setSbtIndexOffset(i) ; 
        p->setAABB( n->AABB() ); 
    }
    return so ; 
}


/**
Foundry::makeSolid11 makes 1-Prim with 1-Node
------------------------------------------------
**/

Solid* Foundry::makeSolid11(const char* label, Node nd, const std::vector<float4>* pl  ) 
{
    unsigned numPrim = 1 ; 
    Solid* so = addSolid(numPrim, label);

    unsigned numNode = 1 ; 
    Prim* p = addPrim(numNode); 
    Node* n = addNode(nd, pl ); 
    p->setAABB( n->AABB() ); 


    float extent = p->extent(); 
    if(extent == 0.f )
    {
        std::cout << "Foundry::makeSolid11 WARNING got zero extent, set to 100.f " << std::endl ; 
        extent = 100.f ; 
    }
    so->extent = extent  ; 
    std::cout << "Foundry::makeSolid11 so->extent " << extent << std::endl ; 

    return so ; 
}

Solid* Foundry::makeSphere(const char* label, float radius)
{
    Node nd = Node::Sphere(radius); 
    return makeSolid11(label, nd, nullptr ); 
}

Solid* Foundry::makeZSphere(const char* label, float radius, float z1, float z2)
{
    Node nd = Node::ZSphere(radius, z1, z2); 
    return makeSolid11(label, nd, nullptr ); 
}

Solid* Foundry::makeCone(const char* label, float r1, float z1, float r2, float z2)
{
    Node nd = Node::Cone(r1, z1, r2, z2 ); 
    return makeSolid11(label, nd, nullptr ); 
}

Solid* Foundry::makeHyperboloid(const char* label, float r0, float zf, float z1, float z2)
{
    Node nd = Node::Hyperboloid( r0, zf, z1, z2 ); 
    return makeSolid11(label, nd, nullptr ); 
}

Solid* Foundry::makeBox3(const char* label, float fx, float fy, float fz )
{
    Node nd = Node::Box3(fx, fy, fz); 
    return makeSolid11(label, nd, nullptr ); 
}

Solid* Foundry::makePlane(const char* label, float nx, float ny, float nz, float d)
{
    Node nd = Node::Plane(nx, ny, nz, d ); 
    return makeSolid11(label, nd, nullptr ); 
}

Solid* Foundry::makeSlab(const char* label, float nx, float ny, float nz, float d1, float d2 )
{
    Node nd = Node::Slab( nx, ny, nz, d1, d1 ); 
    return makeSolid11(label, nd, nullptr ); 
}

Solid* Foundry::makeCylinder(const char* label, float px, float py, float radius, float z1, float z2)
{
    Node nd = Node::Cylinder( px, py, radius, z1, z2 ); 
    return makeSolid11(label, nd, nullptr ); 
}

Solid* Foundry::makeDisc(const char* label, float px, float py, float ir, float r, float z1, float z2)
{
    Node nd = Node::Disc(px, py, ir, r, z1, z2 ); 
    return makeSolid11(label, nd, nullptr ); 
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

Solid* Foundry::makeConvexPolyhedronCube(const char* label, float extent)
{
    float hx = extent ; 
    float hy = extent/2.f ; 
    float hz = extent/3.f ; 

    std::vector<float4> pl ; 
    pl.push_back( make_float4(  1.f,  0.f,  0.f, hx ) ); 
    pl.push_back( make_float4( -1.f,  0.f,  0.f, hx ) ); 
    pl.push_back( make_float4(  0.f,  1.f,  0.f, hy ) ); 
    pl.push_back( make_float4(  0.f, -1.f,  0.f, hy ) ); 
    pl.push_back( make_float4(  0.f,  0.f,  1.f, hz ) ); 
    pl.push_back( make_float4(  0.f,  0.f, -1.f, hz ) );

    Node nd = {} ;
    nd.setAABB(-hx, -hy, -hz, hx, hy, hz); 
    return makeSolid11(label, nd, &pl ); 
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

Solid* Foundry::makeConvexPolyhedronTetrahedron(const char* label, float extent)
{
    //extent = 100.f*sqrt(3); 
    float s = extent ; 

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
    nd.setAABB(extent); 
    return makeSolid11(label, nd, &pl ); 
}








void Foundry::write(const char* base, const char* rel) const 
{
    std::stringstream ss ;   
    ss << base << "/" << rel ; 
    std::string dir = ss.str();   

    std::cout << "Foundry::write " << dir << std::endl ; 

    NP::Write(dir.c_str(), "solid.npy",    (int*)solid.data(), solid.size(), 4 ); 
    NP::Write(dir.c_str(), "prim.npy",   (float*)prim.data(), prim.size(), 4, 3 ); 
    NP::Write(dir.c_str(), "node.npy",   (float*)node.data(), node.size(), 4, 4 ); 
    NP::Write(dir.c_str(), "plan.npy",   (float*)plan.data(), plan.size(), 4 ); 
    NP::Write(dir.c_str(), "tran.npy",   (float*)tran.data(), tran.size(), 4, 4 ); 
}

void Foundry::upload()
{
    unsigned num_solid = solid.size(); 
    unsigned num_prim = prim.size(); 
    unsigned num_node = node.size(); 
    unsigned num_plan = plan.size(); 
    unsigned num_tran = tran.size(); 

    std::cout 
        << "Foundry::upload"
        << " num_solid " << num_solid
        << " num_prim " << num_prim
        << " num_node " << num_node
        << " num_plan " << num_plan
        << " num_tran " << num_tran
        << std::endl
        ;

    d_solid = num_solid > 0 ? CU::UploadArray<Solid>(solid.data(), num_solid ) : nullptr ; 
    d_prim = num_prim > 0 ? CU::UploadArray<Prim>(prim.data(), num_prim ) : nullptr ; 
    d_node = num_node > 0 ? CU::UploadArray<Node>(node.data(), num_node ) : nullptr ; 
    d_plan = num_plan > 0 ? CU::UploadArray<float4>(plan.data(), num_plan ) : nullptr ; 
    d_tran = num_tran > 0 ? CU::UploadArray<qat4>(tran.data(), num_tran ) : nullptr ; 
}

