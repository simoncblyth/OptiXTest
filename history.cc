// name=history ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>

#include "error.h"
#include "history.h"

int main(int argc, char** argv)
{
    History hist ; 
    hist.curr = -1 ; 
    hist.ctrl[0] = 0 ;
    hist.ctrl[1] = 0 ;
    hist.idx[0] = 0 ;
    hist.idx[1] = 0 ;

    for(unsigned nodeIdx=1 ; nodeIdx < 32 ; nodeIdx++)
    {
        int ctrl = nodeIdx ; 
        history_append( hist, nodeIdx, ctrl ); 

        std::cout 
             << " " << std::setw(16) << std::hex << hist.idx[1]
             << " " << std::setw(16) << std::hex << hist.idx[0]  
             << " " << std::setw(16) << std::hex << hist.ctrl[1]
             << " " << std::setw(16) << std::hex << hist.ctrl[0]  
             << std::endl 
             ;      

    }

    return 0 ; 
}




