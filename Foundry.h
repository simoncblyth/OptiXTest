#pragma once

#include <vector>

// maybe Foundry should operate only at node/plan/tran level : eliminating solid and prim  from here ?
#include "Solid.h"
#include "Prim.h"

#include "Node.h"
#include "Quad.h"
#include "qat4.h"


struct Foundry
{
    Foundry();
    void init(); 

    void makeSolids() ;
    void dump() const ;


    unsigned getNumSolid() const;  // ?
    unsigned getNumPrim() const;   // ?

    const Solid* getSolid(const char* name) const ;
    const Solid*   getSolid(unsigned solidIdx) const ;  // ?
    const Prim*    getPrim(unsigned primIdx) const ;    // ?




    unsigned getNumNode() const;
    unsigned getNumPlan() const;
    unsigned getNumTran() const;

    const Node*    getNode(unsigned nodeIdx) const ;
    const float4*  getPlan(unsigned planIdx) const ;
    const qat4*    getTran(unsigned tranIdx) const ;

    unsigned make(const char* name); 

    void addNode(Node& nd, bool prim_check, const std::vector<float4>* pl );
    void addNodes(const std::vector<Node>& nds, bool prim_check );
    void addPrim(int num_node);  // ?


    unsigned makeLayeredSphere(const char* label, float outer_radius, unsigned layers ) ;

    unsigned makeSolid11(float extent, const char* label, Node& nd, const std::vector<float4>* pl  );
    unsigned makeSphere(     const char* label="sphere",      float r=100.f ); 
    unsigned makeZSphere(    const char* label="zsphere",     float r=100.f,  float z1=-50.f , float z2=50.f ); 
    unsigned makeCone(       const char* label="cone",        float r1=300.f, float z1=-300.f, float r2=100.f,   float z2=-100.f ); 
    unsigned makeHyperboloid(const char* label="hyperboloid", float r0=100.f, float zf=50.f,   float z1=-50.f,   float z2=50.f );
    unsigned makeBox3(       const char* label="box3",        float fx=100.f, float fy=200.f,  float fz=300.f );
    unsigned makePlane(      const char* label="plane",       float nx=1.0f,  float ny=0.f,    float nz=0.f,     float d=0.f );
    unsigned makeSlab(       const char* label="slab",        float nx=1.0f,  float ny=0.f,    float nz=0.f,     float d1=-10.f, float d2=10.f );
    unsigned makeCylinder(   const char* label="cylinder",    float px=0.f,   float py=0.f,    float r=100.f,    float z1=-50.f, float z2=50.f );
    unsigned makeDisc(       const char* label="disc",        float px=0.f,   float py=0.f,    float ir=50.f,    float r=100.f,  float z1=-2.f, float z2=2.f);

    static float4 TriPlane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k );
    unsigned makeConvexPolyhedronCube(const char* label="convexpolyhedron_cube");
    unsigned makeConvexPolyhedronTetrahedron(const char* label="convexpolyhedron_tetrahedron");

    void upload();


    std::vector<Solid>  solid ;  
    std::vector<Prim>   prim ; 
  
    // maybe Foundry should operate only at node/plan/tran level : eliminating solid and prim  from here ?

    std::vector<Node>   node ; 
    std::vector<float4> plan ; 
    std::vector<qat4>   tran ;  

    Node*    d_node ; 
    float4*  d_plan ; 
    qat4*    d_tran ; 


};


