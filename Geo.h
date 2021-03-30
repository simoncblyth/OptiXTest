#pragma once
#include <array>
#include <vector>
#include <string>

struct Shape ; 
struct Grid ; 
struct Foundry ; 

/**
Can Foundry replace Geo ?

* no foundry is for now just for making local solids 
  and holding the constituent Node, Prim etc..

**/

struct Geo
{
    static Geo* fGeo ; 
    static Geo* Get();  

    Geo();

    void init();
    void init_sphere_containing_grid_of_spheres(float& tminf, float& tmaxf, unsigned layers);
    void init_sphere(float& tminf, float& tmaxf, unsigned layers);
    void init_zsphere(float& tminf, float& tmaxf, unsigned layers);
    std::string desc() const ;

    unsigned getNumShape() const ; 
    unsigned getNumGrid() const ; 

    const Shape* getShape(int shape_idx_) const ; 
    const Grid*  getGrid(int grid_idx_) const ; 

    void addShape(const char* typs, float outer_extent, unsigned layers);
    void addShape(const char* typs, const std::vector<float>& extents);
    void addGrid(const Grid* grid) ;

    void write(const char* prefix) const ; 
    void setTopExtent(float top_extent_); 
    float getTopExtent() const ; 


    int kludge_outer_aabb = 0 ; 
    int gas_bi_aabb = 0 ; 
    float tmin = 0.f ; 
    float tmax = 1e16f ; 
    float top_extent = 100.f ; 

    Foundry*                  foundry ; 
    std::vector<const Shape*> shapes ; 
    std::vector<const Grid*>  grids ; 
    const char*               top ;  


};


