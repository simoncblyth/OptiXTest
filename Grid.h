#pragma once

#include <array>
#include <vector>
#include <glm/glm.hpp>

struct Grid
{ 
    unsigned               ias_idx ; 
    unsigned               num_solid ; 
    float                  gridscale ; 

    std::array<int,9>      grid ; 
    std::vector<unsigned>  solid_modulo ;  
    std::vector<unsigned>  solid_single ;  
    std::vector<glm::mat4> trs ;  

    Grid(unsigned ias_idx, unsigned num_solid);

    const float4 center_extent() const ;
    std::string desc() const ;

    void init(); 
    void write(const char* base, const char* rel, unsigned idx ) const ;
};


