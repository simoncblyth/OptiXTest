// name=csg_classify ; gcc $name.cc -std=c++11 -DDEBUG=1 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>

#include "OpticksCSG.h"
#include "csg_classify.h"

int main(int argc, char** argv)
{
    LUT lut ;  

    for(int o=0 ; o < 3 ; o++)
    {
        OpticksCSG_t operation = (OpticksCSG_t)(o+1) ; 
        const char* opname = CSG::Name(operation) ;
        std::cout << std::endl ; 

        for(int c=0 ; c < 2 ; c++)
        {
            bool ACloser = (bool)c ; 
            std::cout 
                << " " << opname << " with " << ( ACloser ? "A" : "B" ) << " closer " << std::endl 
                ; 


            std::cout 
                << std::setw(10) << "A"
                << std::setw(10) << "B"
                << std::setw(10) << "Action"
                << std::endl 
                ;

            for(int a=0 ; a < 3 ; a++)
            for(int b=0 ; b < 3 ; b++)
            {

                IntersectionState_t stateA = (IntersectionState_t)a ; 
                IntersectionState_t stateB = (IntersectionState_t)b ; 

                int action = lut.lookup( operation, stateA, stateB, ACloser );

                std::cout 
                    << std::setw(10) << IntersectionState::Name(stateA)
                    << std::setw(10) << IntersectionState::Name(stateB)
                    << std::setw(20) << CTRL::Name(action)
                    << std::endl 
                    ; 

            }
        }
    }
    
    return 0 ; 
}
