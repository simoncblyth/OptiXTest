#pragma once
#include <string>

struct Solid   // Composite shape 
{
    Solid( const char* label,  int numPrim, int primOffset, float extent ); 

    char        label[4] ; 
    int         numPrim ; 
    int         primOffset ; 
    float       extent ; 

    std::string desc() const ; 

};


