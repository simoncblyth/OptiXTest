// ./UtilTest.sh 
#include <iostream>
#include <string>
#include <array>
#include "Util.h"


void test_ParseGridSpec(int argc, char** argv)
{
    std::string clusterspec = Util::GetEValue<std::string>("CLUSTERSPEC","-1:2:1,-1:2:1,-1:2:1") ; 
    const char* spec = argc > 1 ? argv[1] : clusterspec.c_str(); 
    std::cout << "clusterspec " << spec << std::endl; 
    std::array<int,9> cl ; 
    Util::ParseGridSpec(cl, spec); // string parsed into array of 9 ints 
    Util::DumpGrid(cl); 
}

void test_Encode4()
{
    const char* s = "abcd" ; 
    unsigned u4 = Util::Encode4(s); 
    std::cout << " s " << s << " u4 " << u4 << std::endl ; 
}


int main(int argc, char** argv)
{
    //test_parseGridSpec(argc, argv); 
    test_Encode4(); 
  
    return 0 ; 
}
