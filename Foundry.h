#pragma once

#include <vector>

#include "Prim.h"
#include "Node.h"
#include "Solid.h"
#include "Quad.h"


struct Foundry
{
    void makeSolids() ;

    Solid* make(const char* name); 
    void addNode(Node& nd, bool prim_check, const std::vector<float4>* pl );
    void addNodes(const std::vector<Node>& nds, bool prim_check );
    void addPrim(int num_node);

    Solid* makeSolid11(float extent, const char* label, Node& nd, const std::vector<float4>* pl  );

    Solid* makeSphere(float radius=100.f); 
    Solid* makeZSphere(); 
    Solid* makeCone(); 
    Solid* makeHyperboloid();
    Solid* makeBox3();
    Solid* makePlane();
    Solid* makeSlab();
    Solid* makeCylinder();
    Solid* makeDisc();

    static float4 TriPlane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k );
    Solid* makeConvexPolyhedronCube();
    Solid* makeConvexPolyhedronTetrahedron();

    std::vector<Prim>   prim ; 
    std::vector<Node>   node ; 
    std::vector<float4> plan ; 
    std::vector<quad4>  tran ; 
    std::vector<Solid>  solid ; 

};


