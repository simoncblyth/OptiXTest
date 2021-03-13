// name=InstanceIdTest ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include <cassert>

#include "InstanceId.h"

int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 

    unsigned ins_idx_(0) ; 
    unsigned gas_idx_(0) ; 
    unsigned count(0) ; 

    std::cout 
        << " InstanceId::ins_bits " << std::dec << InstanceId::ins_bits
        << " InstanceId::ins_mask " << std::hex << InstanceId::ins_mask 
        << std::endl 
        << " InstanceId::gas_bits " << std::dec << InstanceId::gas_bits
        << " InstanceId::gas_mask " << std::hex << InstanceId::gas_mask
        << std::endl 
        ;

    for(unsigned ins_idx=0 ; ins_idx < InstanceId::ins_mask ; ins_idx++){ 
    for(unsigned gas_idx=0 ; gas_idx < InstanceId::gas_mask ; gas_idx++){  

       unsigned id = InstanceId::Encode(ins_idx, gas_idx ); 
       InstanceId::Decode(ins_idx_, gas_idx_, id ); 

       bool gas_match =  gas_idx == gas_idx_ ;
       bool ins_match = ins_idx == ins_idx_ ;

       if( count % 1000000 == 0 || !gas_match || !ins_match )
       std::cout 
           << " count "     << std::setw(10) << std::dec << count
           << " ins_idx "   << std::setw(10) << std::hex << ins_idx 
           << " ins_idx_ "  << std::setw(10) << std::hex << ins_idx_ 
           << " gas_idx "   << std::setw(10) << std::hex << gas_idx 
           << " gas_idx_ "  << std::setw(10) << std::hex << gas_idx_
           << " id "        << std::setw(10) << std::hex << id
           << std::endl
           ;
        
       assert( gas_match ); 
       assert( ins_match ); 

       count++ ; 
    }
    }
    return 0 ; 
}


