#include <iostream>
#include <iomanip>

#include "Tran.h"

int main(int argc, char** argv)
{
    const Tran<double>* tr = Tran<double>::make_translate( 100., 200., 300. ); 
    std::cout << "tr" << *tr << std::endl ;  

    const Tran<double>* sc = Tran<double>::make_scale( 1., 1., 0.5 ); 
    std::cout << "sc" << *sc << std::endl ;  

    const Tran<double>* ro = Tran<double>::make_rotate(0., 0., 1., 45.); 
    std::cout << "ro" << *ro << std::endl ;  

    const Tran<double>* srt = Tran<double>::product( sc, ro, tr, true );  
    std::cout << "srt" << *srt << std::endl ;  

    return 0 ; 
}
