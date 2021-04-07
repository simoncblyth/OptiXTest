#pragma once
#include <array>
#include <vector>
#include <string>


#include "PrimSpec.h"

struct Solid ; 
struct Prim ; 
struct Grid ; 
struct Foundry ; 

/**
Can Foundry replace Geo ?

* no foundry is for now just for making local solids 
  and holding the constituent Node, Prim etc..

**/

struct Geo
{
    Geo(Foundry* foundry_);

    void init();
    void init_sphere_containing_grid_of_spheres(float& tminf, float& tmaxf, unsigned layers);
    void init_parade(float& tminf, float& tmaxf );
    void init_layered(const char* name, float& tminf, float& tmaxf, unsigned layers);
    void init_clustered(const char* name, float& tminf, float& tmaxf );

    void init(const char* name, float& tminf, float& tmaxf);
    std::string desc() const ;

    unsigned getNumSolid() const ; 
    unsigned getNumPrim() const ; 

    const Solid* getSolid(unsigned solidIdx) const ; 
    PrimSpec getPrimSpec(unsigned solidIdx) const ;
    const Prim* getPrim(unsigned primIdx) const ; 

    unsigned getNumGrid() const ; 
    const Grid*  getGrid(unsigned gridIdx) const ; 
    const Grid*  getGrid_(int gridIdx) const ; 

    void addGrid(const Grid* grid) ;

    void write(const char* prefix) const ; 
    void setCenterExtent(const float4& center_extent); 
    float4 getCenterExtent() const ; 

    float tmin = 0.f ; 
    float tmax = 1e16f ; 
    float4 center_extent = {0.f, 0.f, 0.f, 100.f} ; 

    Foundry*                  foundry ; 
    std::vector<const Grid*>  grids ; 
    const char*               top ;  

};


