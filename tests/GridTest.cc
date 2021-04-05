// ./GridTest.sh 
#include <iostream>
#include "Grid.h"

int main(int argc, char** argv)
{
    unsigned ias_idx = 0 ; 
    unsigned num_solid = 3 ;  
    Grid g0(ias_idx, num_solid); 
    std::cout << g0.desc() << std::endl ;  
    g0.write("/tmp", "GridTestWrite", 0 ); 

    return 0 ; 
}
