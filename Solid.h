#pragma once

#include <string>

struct Solid   // Composite shape 
{
    const char* label ; 

    int         numPrim ; 
    int         primOffset ; 
    float       extent ;  // TODO: can this be replaced ?  perhaps use a slot in the node ? need to sort out AABB 

    std::string desc() const ; 

};


