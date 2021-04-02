#include <string>
#include <sstream>
#include <iostream>
#include "NP.hh"
#include "Sys.h"
#include "Util.h"
#include "Grid.h"
#include "InstanceId.h"

#include <glm/gtx/transform.hpp>


/**
Geo::makeGrid
---------------

shape_modulo
    vector of gas_idx which are modulo cycled in a 3d grid array  

shape_single
    vector of gas_idx which are singly included into the IAS 
    with an identity transform

Create vector of transfoms and creat IAS from that.
Currently a 3D grid of translate transforms with all available GAS repeated modulo

**/


Grid::Grid( unsigned ias_idx_, unsigned num_solid_ )
    :
    ias_idx(ias_idx_),
    num_solid(num_solid_),
    gridscale(Util::GetEValue<float>("GRIDSCALE", 1.f))
{
    std::string gridspec = Util::GetEValue<std::string>("GRIDSPEC","-10:11,2,-10:11:2,-10:11,2") ; 
    Util::ParseGridSpec(grid, gridspec.c_str());      // string parsed into array of 9 ints 
    Util::GetEVector(solid_modulo, "GRIDMODULO", "0,1" ); 
    Util::GetEVector(solid_single, "GRIDSINGLE", "2" ); 

    std::cout << "GRIDSPEC " << gridspec << std::endl ; 
    std::cout << "GRIDSCALE " << gridscale << std::endl ; 
    std::cout << "GRIDMODULO " << Util::Present(solid_modulo) << std::endl ; 
    std::cout << "GRIDSINGLE " << Util::Present(solid_single) << std::endl ; 

    init(); 
}

float Grid::extent() const 
{
    int mn(0); 
    int mx(0); 
    Util::GridMinMax(grid, mn, mx); 
   
    int iextent = std::max( std::abs(mn), std::abs(mx) );  // half side
    return gridscale*float(iextent) ; 
}

std::string Grid::desc() const 
{
    std::stringstream ss ; 
    ss << "Grid extent " << extent() << " num_tr " << trs.size() ; 
    std::string s = ss.str(); 
    return s; 
}


void Grid::init()
{
    unsigned num_solid_modulo = solid_modulo.size() ; 
    unsigned num_solid_single = solid_single.size() ; 

    // check the input solid_idx are valid 
    for(unsigned i=0 ; i < num_solid_modulo ; i++ ) assert(solid_modulo[i] < num_solid) ; 
    for(unsigned i=0 ; i < num_solid_single ; i++ ) assert(solid_single[i] < num_solid) ; 

    std::cout 
        << "Grid::init"
        << " num_solid_modulo " << num_solid_modulo
        << " num_solid_single " << num_solid_single
        << " num_solid " << num_solid
        << std::endl
        ;

    for(int i=0 ; i < int(num_solid_single) ; i++)
    {
        unsigned ins_idx = trs.size() ;        // 0-based index within the Grid
        unsigned gas_idx = solid_single[i] ;   // 0-based solid index
        unsigned id = InstanceId::Encode( ins_idx, gas_idx ); 

        glm::mat4 tr(1.f) ;  // identity transform for the large sphere 
        tr[0][3] = Sys::unsigned_as_float(id); 
        tr[1][3] = Sys::unsigned_as_float(0) ;
        tr[2][3] = Sys::unsigned_as_float(0) ;   
        tr[3][3] = Sys::unsigned_as_float(0) ;   

        trs.push_back(tr); 
    }

    for(int i=grid[0] ; i < grid[1] ; i+=grid[2] ){
    for(int j=grid[3] ; j < grid[4] ; j+=grid[5] ){
    for(int k=grid[6] ; k < grid[7] ; k+=grid[8] ){

        glm::vec3 tlat(i*gridscale,j*gridscale,k*gridscale) ;  // grid translation 
        glm::mat4 tr(1.f) ;
        tr = glm::translate(tr, tlat );

        unsigned ins_idx = trs.size() ;     
        unsigned solid_modulo_idx = ins_idx % num_solid_modulo ; 
        unsigned gas_idx = solid_modulo[solid_modulo_idx] ; 
        unsigned id = InstanceId::Encode( ins_idx, gas_idx ); 

        tr[0][3] = Sys::unsigned_as_float(id); 
        tr[1][3] = Sys::unsigned_as_float(0) ;
        tr[2][3] = Sys::unsigned_as_float(0) ;   
        tr[3][3] = Sys::unsigned_as_float(0) ;   

        trs.push_back(tr); 
    }
    }
    }
}


void Grid::write(const char* base, const char* rel, unsigned idx ) const 
{
    std::stringstream ss ;   
    ss << base << "/" << rel << "/" << idx << "/" ; 
    std::string dir = ss.str();   

    std::cout 
        << "Grid::write "
        << " trs.size " << trs.size()
        << " dir " << dir
        << std::endl 
        ;

    NP::Write(dir.c_str(), "grid.npy", (float*)trs.data(),  trs.size(), 4, 4 ); 
}


