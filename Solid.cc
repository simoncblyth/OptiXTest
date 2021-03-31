#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "Solid.h"


Solid::Solid( const char* label_,  int numPrim_, int primOffset_, float extent_ )
    :
    numPrim(numPrim_),
    primOffset(primOffset_), 
    extent(extent_)
{
    memcpy(label, label_, 4 );
}


std::string Solid::desc() const 
{
    std::stringstream ss ; 
    ss << "Solid " 
       << std::setw(30) << label 
       << " numPrim " << std::setw(3) << numPrim 
       << " primOffset " << std::setw(3) << primOffset
       << " extent " << std::setw(10) << extent
       ; 
    std::string s = ss.str(); 
    return s ; 
}

