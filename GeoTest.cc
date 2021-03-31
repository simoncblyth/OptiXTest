// ./GeoTest.sh

#include "sutil_vec_math.h"

#include <iostream>
#include "Foundry.h"
#include "Geo.h"

int main(int argc, char** argv)
{
    Foundry foundry ; 
    Geo geo(&foundry) ; 
    std::cout << geo.desc() << std::endl ; 

    geo.write("/tmp/GeoTest_" ); 

    return 0 ; 
}

