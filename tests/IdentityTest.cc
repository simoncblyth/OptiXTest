// ./IdentityTest.sh

#include <iostream>
#include <iomanip>
#include <cassert>
#include "Identity.h"

void test_scan()
{
    std::cout << " test_scan " << std::endl ; 
    unsigned ias_idx_(0) ;
    unsigned ins_idx_(0) ; 
    unsigned gas_idx_(0) ; 
    unsigned count(0) ; 

    for(unsigned ias_idx=0 ; ias_idx <  0xf    ; ias_idx++){
    for(unsigned ins_idx=0 ; ins_idx <  0xffff ; ins_idx++){
    for(unsigned gas_idx=0 ; gas_idx <  0xfff  ; gas_idx++){

       unsigned id = Identity::Encode(ias_idx, ins_idx, gas_idx ); 
       Identity::Decode(ias_idx_, ins_idx_, gas_idx_, id ); 

       if( count % 10000000 == 0 )
       std::cout 
           << " count "    << std::setw(10) << count
           << " ias_idx "  << std::setw(10) << ias_idx 
           << " ins_idx "  << std::setw(10) << ins_idx 
           << " gas_idx "  << std::setw(10) << gas_idx 
           << " id "       << std::setw(10) << id
           << std::endl
           ;
        
       assert( gas_idx == gas_idx_ ); 
       assert( ias_idx == ias_idx_ ); 
       assert( ins_idx == ins_idx_ ); 

       count++ ; 
    }
    }
    }
}

void test_zero()
{
    std::cout << " test_zero " << std::endl ; 
    unsigned ias_idx = 0u ; 
    unsigned ins_idx = 0u ; 
    unsigned gas_idx = 0u ; 

    unsigned id = Identity::Encode(ias_idx,ins_idx,gas_idx); 
    std::cout << " id " << id << std::endl ; 
    unsigned ias_idx_ ; 
    unsigned ins_idx_ ; 
    unsigned gas_idx_ ; 
    Identity::Decode(ias_idx_, ins_idx_, gas_idx_, id );

    std::cout 
        << " id " << id 
        << " ias_idx_ " << ias_idx_ 
        << " ins_idx_ " << ins_idx_ 
        << " gas_idx_ " << gas_idx_ 
        << std::endl
        ; 
    assert( ias_idx_ == ias_idx ); 
    assert( ins_idx_ == ins_idx ); 
    assert( gas_idx_ == gas_idx ); 
    assert( id > 0u );   
}

int main(int argc, char** argv)
{
    test_zero(); 
    test_scan(); 
    return 0 ; 
}




