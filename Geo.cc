#include <iostream>
#include <iomanip>
#include <cstring>
#include <vector_types.h>

#include "Sys.h"
#include "Util.h"
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"

#include "NP.hh"

#include "sutil_vec_math.h"

#include "Prim.h"
#include "Foundry.h"
#include "Geo.h"
#include "Grid.h"
#include "InstanceId.h"

Geo::Geo(Foundry* foundry_)
    :
    foundry(foundry_),
    top("i0")
{
    init();
}

void Geo::init()
{
    float tminf(0.1) ; 
    float tmaxf(10000.f) ; 

    std::string geometry = Util::GetEValue<std::string>("GEOMETRY", "sphere_containing_grid_of_spheres" ); 
    unsigned layers = Util::GetEValue<unsigned>("LAYERS", 1) ; 

    std::cout
        << "Geo::init"
        << " geometry " << geometry
        << " layers " << layers 
        << std::endl 
        ;    

    if(strcmp(geometry.c_str(), "sphere_containing_grid_of_spheres") == 0)
    {
        init_sphere_containing_grid_of_spheres(tminf, tmaxf, layers );
    }
    else if(strcmp(geometry.c_str(), "sphere") == 0 )
    {
        init_sphere(tminf, tmaxf, layers);
    }
    else if(strcmp(geometry.c_str(), "zsphere") == 0 )
    {
        init_zsphere(tminf, tmaxf, layers);
    }
    else
    {
        assert(0); 
    }

    float top_extent = getTopExtent(); 
    tmin = top_extent*tminf ; 
    tmax = top_extent*tmaxf ; 
    std::cout 
        << "Geo::init" 
        << " top_extent " << top_extent  
        << " tminf " << tminf 
        << " tmin " << tmin 
        << " tmaxf " << tmaxf 
        << " tmax " << tmax 
        << std::endl 
        ; 

    float e_tminf = Util::GetEValue<float>("TMIN", -1.0) ; 
    if(e_tminf > 0.f )
    {
        tmin = top_extent*e_tminf ; 
        std::cout << "Geo::init e_tminf TMIN " << e_tminf << " override tmin " << tmin << std::endl ; 
    }
    
    float e_tmaxf = Util::GetEValue<float>("TMAX", -1.0) ; 
    if(e_tmaxf > 0.f )
    {
        tmax = top_extent*e_tmaxf ; 
        std::cout << "Geo::init e_tmaxf TMAX " << e_tmaxf << " override tmax " << tmax << std::endl ; 
    }
}

/**
Geo::init_sphere_containing_grid_of_spheres
---------------------------------------------

A cube of side 1 (halfside 0.5) has diagonal sqrt(3):1.7320508075688772 
that will fit inside a sphere of diameter sqrt(3) (radius sqrt(3)/2 : 0.86602540378443)
Container sphere "extent" needs to be sqrt(3) larger than the grid extent.

**/

void Geo::init_sphere_containing_grid_of_spheres(float& tminf, float& tmaxf, unsigned layers )
{
    std::cout << "Geo::init_sphere_containing_grid_of_spheres : layers " << layers << std::endl ; 

    unsigned ias_idx = grids.size(); 
    unsigned num_shape = 3 ; 
    Grid* grid = new Grid(ias_idx, num_shape) ; 
    addGrid(grid); 

    float big_radius = float(grid->extent())*sqrtf(3.f) ;
    std::cout << " big_radius " << big_radius << std::endl ; 

    foundry->makeLayered("sphere", 0.7f, layers ); 
    foundry->makeLayered("sphere", 1.0f, layers ); 
    foundry->makeLayered("sphere", big_radius, 1 ); 

    top = strdup("i0") ; 

    setTopExtent(big_radius); 

    tminf = 0.75f ; 
    tmaxf = 10000.f ; 
}

void Geo::init_sphere(float& tminf, float& tmaxf, unsigned layers)
{
    std::cout << "Geo::init_sphere" << std::endl ; 

    foundry->makeLayered("sphere", 100.f, layers ); 

    setTopExtent(100.f); 
    top = strdup("g0") ; 

    tminf = 1.60f ;   //  hmm depends on viewpoint, aiming to cut into the sphere with the tmin
    tmaxf = 10000.f ; 
}

void Geo::init_zsphere(float& tminf, float& tmaxf, unsigned layers)
{
    std::cout << "Geo::init_zsphere" << std::endl ; 

    foundry->makeLayered("zsphere", 100.f, layers ); 

    setTopExtent(100.f); 
    top = strdup("g0") ; 

    tminf = 1.60f ;   //  hmm depends on viewpoint, aiming to cut into the sphere with the tmin
    tmaxf = 10000.f ; 
}


std::string Geo::desc() const
{
    std::stringstream ss ; 
    ss << "Geo " << " grids:" << grids.size() ; 
    std::string s = ss.str(); 
    return s ; 
}

void Geo::setTopExtent(float top_extent_){ top_extent = top_extent_ ;  }
float Geo::getTopExtent() const  { return top_extent ;  }


unsigned        Geo::getNumSolid() const {                        return foundry->getNumSolid() ;      }
const Solid*    Geo::getSolid(         unsigned solidIdx) const { return foundry->getSolid(solidIdx);  }
PrimSpec        Geo::getPrimSpec(      unsigned solidIdx) const { return foundry->getPrimSpec(solidIdx);  }
const Prim*     Geo::getPrim(          unsigned primIdx) const  { return foundry->getPrim(primIdx);  }




void Geo::addGrid(const Grid* grid)
{
    grids.push_back(grid); 
}
unsigned Geo::getNumGrid() const 
{
    return grids.size() ; 
}
const Grid* Geo::getGrid_(int gridIdx_) const
{
    unsigned gridIdx = gridIdx_ < 0 ? grids.size() + unsigned(gridIdx_) : unsigned(gridIdx_) ;  
    return getGrid(gridIdx);  
}
const Grid* Geo::getGrid(unsigned gridIdx) const
{
    assert( gridIdx < grids.size() );
    return grids[gridIdx] ; 
}


void Geo::write(const char* dir) const 
{
    std::cout << "Geo::write " << dir << std::endl ;  

    NP spec("<u4", 4); 
    unsigned* v = spec.values<unsigned>() ; 
    *(v+0) = getNumSolid(); 
    *(v+1) = getNumGrid(); 
    *(v+2) = InstanceId::ins_bits ; 
    *(v+3) = InstanceId::gas_bits ; 
    spec.save(dir, "spec.npy"); 

    // with foundry it makes more sense to write everything at once, not individual solids
    foundry->write(dir, "foundry"); 

    unsigned num_grid = getNumGrid(); 
    for(unsigned i=0 ; i < num_grid ; i++) 
    {
        const Grid* gr = getGrid(i); 
        gr->write(dir,"grid", i); 
    }
}
