
#include "sutil_vec_math.h"
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
       << " center_extent " << center_extent
       ; 
    std::string s = ss.str(); 
    return s ; 
}

#endif

