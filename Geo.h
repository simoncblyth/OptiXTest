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
    void init_sphere(float& tminf, float& tmaxf, unsigned layers);
    void init_zsphere(float& tminf, float& tmaxf, unsigned layers);
    std::string desc() const ;

    unsigned getNumSolid() const ; 

    const Solid* getSolid(unsigned solidIdx) const ; 
    PrimSpec getPrimSpec(unsigned solidIdx) const ;
    const Prim* getPrim(unsigned primIdx) const ; 

    unsigned getNumGrid() const ; 
    const Grid*  getGrid(unsigned gridIdx) const ; 
    const Grid*  getGrid_(int gridIdx) const ; 

    void addGrid(const Grid* grid) ;

    void write(const char* prefix) const ; 
    void setTopExtent(float top_extent_); 
    float getTopExtent() const ; 

    float tmin = 0.f ; 
    float tmax = 1e16f ; 
    float top_extent = 100.f ; 

    Foundry*                  foundry ; 
    std::vector<const Grid*>  grids ; 
    const char*               top ;  

};


