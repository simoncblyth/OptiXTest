
#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include "Prim.h"

#include <iostream>
#include <iomanip>
#include <sstream>

std::string Prim::desc() const 
{  
    std::stringstream ss ; 
    ss 
      << "Prim"
      << " numNode "    << std::setw(3) << numNode 
      << " nodeOffset " << std::setw(3) << nodeOffset
      << " tranOffset " << std::setw(3) << tranOffset
      << " planOffset " << std::setw(3) << planOffset
      ;
    
    std::string s = ss.str(); 
    return s ; 
}
#endif


