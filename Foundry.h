#pragma once

#include <vector>

#include "Prim.h"
#include "Node.h"
#include "Solid.h"
#include "Quad.h"


struct Foundry
{
    Foundry();
    void init(); 

    void makeSolids() ;
    void dump() const ;

    unsigned getNumSolid() const;
    unsigned getNumPrim() const;
    unsigned getNumNode() const;
    unsigned getNumPlan() const;
    unsigned getNumTran() const;

    const Solid*   getSolid(unsigned solidIdx) const ;
    const Prim*    getPrim(unsigned primIdx) const ;
    const Node*    getNode(unsigned nodeIdx) const ;
    const float4*  getPlan(unsigned planIdx) const ;
    const quad4*   getTran(unsigned tranIdx) const ;

    const Solid* getSolid(const char* name) const ;

    unsigned make(const char* name); 

    void addNode(Node& nd, bool prim_check, const std::vector<float4>* pl );
    void addNodes(const std::vector<Node>& nds, bool prim_check );
    void addPrim(int num_node);

    unsigned makeSolid11(float extent, const char* label, Node& nd, const std::vector<float4>* pl  );

    unsigned makeSphere(     const char* label="sphere",  float radius=100.f ); 
    unsigned makeZSphere(    const char* label="zsphere"); 
    unsigned makeCone(       const char* label="cone"); 
    unsigned makeHyperboloid(const char* label="hyperboloid");
    unsigned makeBox3(       const char* label="box3");
    unsigned makePlane(      const char* label="plane");
    unsigned makeSlab(       const char* label="slab");
    unsigned makeCylinder(   const char* label="cylinder");
    unsigned makeDisc(       const char* label="disc");

    static float4 TriPlane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k );
    unsigned makeConvexPolyhedronCube(const char* label="convexpolyhedron_cube");
    unsigned makeConvexPolyhedronTetrahedron(const char* label="convexpolyhedron_tetrahedron");

    std::vector<Solid>  solid ; 
    std::vector<Prim>   prim ; 
    std::vector<Node>   node ; 
    std::vector<float4> plan ; 
    std::vector<quad4>  tran ; 

};


