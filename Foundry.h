#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "Solid.h"
#include "Prim.h"
#include "Node.h"
#include "Quad.h"
#include "qat4.h"
#include "Tran.h"

/**
Foundry
========

* Solids contain one or more Prim  (Prim would correspond to Geant4 G4VSolid)
* Prim contain one or more Node    (Node are CSG constituent nodes) 

**/


struct Foundry
{
    Foundry();
    void init(); 

    void makeDemoSolids() ;
    std::string getBashMap();

    void dump() const ;
    void dumpSolid(unsigned solidIdx ) const ;
    void dumpPrim(unsigned solidIdx ) const ;
    void dumpNode(unsigned solidIdx ) const ;

    PrimSpec getPrimSpec(       unsigned solidIdx) const ;
    PrimSpec getPrimSpecHost(   unsigned solidIdx) const ;
    PrimSpec getPrimSpecDevice( unsigned solidIdx) const ;

    const Solid*   getSolid(const char* name) const ;
    const Solid*   getSolid_(int solidIdx) const ;   // -ve counts from back 
    unsigned       getSolidIdx(const Solid* so) const ; 

    unsigned getNumSolid() const; 
    unsigned getNumPrim() const;   
    unsigned getNumNode() const;
    unsigned getNumPlan() const;
    unsigned getNumTran() const;
    unsigned getNumItra() const;

    const Solid*   getSolid(unsigned solidIdx) const ;  
    const Prim*    getPrim(unsigned primIdx) const ;    
    const Node*    getNode(unsigned nodeIdx) const ;
    const float4*  getPlan(unsigned planIdx) const ;
    const qat4*    getTran(unsigned tranIdx) const ;
    const qat4*    getItra(unsigned itraIdx) const ;

    Solid* addSolid(unsigned num_prim, const char* label );
    Prim*  addPrim(int num_node) ;
    Node*  addNode(Node nd, const std::vector<float4>* pl=nullptr );
    Node*  addNodes(const std::vector<Node>& nds );
    float4* addPlan(const float4& pl );

    unsigned addTran( const Tran<double>& tr  );

    Solid* make(const char* name); 
    Solid* makeLayered( const char* label, float outer_radius, unsigned layers ) ;
    Solid* makeClustered(const char* name,  int i0, int i1, int is, int j0, int j1, int js, int k0, int k1, int ks, double unit ) ;

    Solid* makeSolid11(const char* label, Node nd, const std::vector<float4>* pl=nullptr  );

    Solid* makeSphere(     const char* label="sphe", float r=100.f ); 
    Solid* makeEllipsoid(  const char* label="elli", float rx=100.f, float ry=100.f, float rz=50.f ); 
    Solid* makeZSphere(    const char* label="zsph", float r=100.f,  float z1=-50.f , float z2=50.f ); 
    Solid* makeCone(       const char* label="cone", float r1=300.f, float z1=-300.f, float r2=100.f,   float z2=-100.f ); 
    Solid* makeHyperboloid(const char* label="hype", float r0=100.f, float zf=50.f,   float z1=-50.f,   float z2=50.f );
    Solid* makeBox3(       const char* label="box3", float fx=100.f, float fy=200.f,  float fz=300.f );
    Solid* makePlane(      const char* label="plan", float nx=1.0f,  float ny=0.f,    float nz=0.f,     float d=0.f );
    Solid* makeSlab(       const char* label="slab", float nx=1.0f,  float ny=0.f,    float nz=0.f,     float d1=-10.f, float d2=10.f );
    Solid* makeCylinder(   const char* label="cyli", float px=0.f,   float py=0.f,    float r=100.f,    float z1=-50.f, float z2=50.f );
    Solid* makeDisc(       const char* label="disc", float px=0.f,   float py=0.f,    float ir=50.f,    float r=100.f,  float z1=-2.f, float z2=2.f);

    Solid* makeConvexPolyhedronCube(       const char* label="vcub", float extent=100.f );
    Solid* makeConvexPolyhedronTetrahedron(const char* label="vtet", float extent=100.f);



    static float4 TriPlane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k );

    void write(const char* base, const char* rel) const ;
    void upload();

    unsigned            imax ; 

    std::vector<Solid>  solid ;   
    std::vector<Prim>   prim ; 
    std::vector<Node>   node ; 
    std::vector<float4> plan ; 
    std::vector<qat4>   tran ;  
    std::vector<qat4>   itra ;  

    Solid*   d_solid ; 
    Prim*    d_prim ; 
    Node*    d_node ; 
    float4*  d_plan ; 
    qat4*    d_tran ; 
    qat4*    d_itra ; 
};


