// ./SolidTest.sh
#include "Solid.h"
#include "NP.hh"
#include <iostream>

int main(int argc, char** argv)
{
    Solid r = { "red"  , 1, 0, 10.f }; 
    Solid g = { "gre"  , 1, 1, 20.f };
    Solid b = { "blu"  , 1, 2, 30.f }; 
    Solid c = { "cya"  , 1, 3, 40.f }; 
    Solid m = { "mag"  , 1, 4, 50.f }; 
    Solid y = { "yel"  , 1, 5, 60.f }; 
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
