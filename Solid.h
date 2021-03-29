#pragma once

#include <string>
#include <vector>

#include "Prim.h"
#include "Node.h"

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


