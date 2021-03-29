#pragma once

#include <string>
struct Foundry ; 

struct Solid   // Composite shape 
{
    std::string label ; 

    int   numPrim ; 
    int   primOffset ; 
    float extent ;  // TODO: can this be replaced ?  perhaps use a slot in the node ? need to sort out AABB 

    Foundry* foundry ; 

};


