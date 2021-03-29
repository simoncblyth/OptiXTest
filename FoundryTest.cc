// name=FoundryTest ; gcc $name.cc Foundry.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name 

#include "sutil_vec_math.h"
#include "Foundry.h"

int main(int argc, char** argv)
{
   Foundry fd ;  
   fd.makeSolids(); 
   return 0 ; 
}
