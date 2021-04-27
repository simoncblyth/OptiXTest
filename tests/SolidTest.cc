// ./SolidTest.sh

#include "sutil_vec_math.h"

#include "Solid.h"
#include "NP.hh"
#include <iostream>

int main(int argc, char** argv)
{
    Solid r = { "red"  , 1, 0,  0 }; 
    Solid g = { "gre"  , 1, 1,  0 };
    Solid b = { "blu"  , 1, 2,  0 }; 
    Solid c = { "cya"  , 1, 3,  0 }; 
    Solid m = { "mag"  , 1, 4,  0 }; 
    Solid y = { "yel"  , 1, 5,  0 }; 

    std::vector<Solid> so ; 

    so.push_back(r); 
    so.push_back(g); 
    so.push_back(b); 
    so.push_back(c); 
    so.push_back(m); 
    so.push_back(y); 

    NP::Write( "/tmp", "SolidTest.npy", (int*)so.data(), so.size(), 4 ) ;
    return 0 ; 
}
