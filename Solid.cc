#include <sstream>
#include <iostream>
#include <iomanip>

#include "Solid.h"

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

