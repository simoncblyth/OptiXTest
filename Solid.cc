#include "Solid.h"

#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>


/*

Solid::Solid()
    :
    numPrim(0),
    primOffset(0),
    extent(0.f)
{
   label[0] = '\0' ; 
   label[1] = '\0' ; 
   label[2] = '\0' ; 
   label[3] = '\0' ; 
}

Solid::Solid( const char* label_,  int numPrim_, int primOffset_, float extent_ )
    :
    numPrim(numPrim_),
    primOffset(primOffset_), 
    extent(extent_)
{
    memcpy(label, label_, 4 );
}

*/


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

