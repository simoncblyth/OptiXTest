#include "Solid.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>


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

#endif

