#pragma once
#include <cassert>

/**
Identity
===========

ias_idx
   <= 0x000f (15)
     
ins_idx
   <= 0xffff (65,535) 

gas_idx 
   <  0x0fff (4,095)  "less than" as it gets encoded 1-based to reserve zero  
   

Note that the 0-based gas_idx is encoded as a 1-based "gas_id" 
in order to avoid idenity of zero. 
This is because it is convenient to reserve zero as meaning no-hit.

**/

struct Identity
{
    static void     Decode(unsigned& ias_idx, unsigned& ins_idx, unsigned& gas_idx, const unsigned identity );
    static unsigned Encode(unsigned  ias_idx, unsigned  ins_idx, unsigned  gas_idx );
};

inline void Identity::Decode(unsigned& ias_idx, unsigned& ins_idx, unsigned& gas_idx, const unsigned identity ) // static 
{
    ias_idx = (( 0xf0000000 & identity ) >> 28 ) - 0u ; 
    ins_idx = (( 0x0ffff000 & identity ) >> 12 ) - 0u ; 
    gas_idx = (( 0x00000fff & identity ) >>  0 ) - 1u ;  
}
inline unsigned Identity::Encode(unsigned ias_idx, unsigned ins_idx, unsigned gas_idx ) // static 
{
    assert( ias_idx <= 0xf ); 
    assert( ins_idx <= 0xffff ); 
    assert( gas_idx <  0xfff );   

    unsigned identity = 
        (( (0u + ias_idx) << 28 ) & 0xf0000000 ) |
        (( (0u + ins_idx) << 12 ) & 0x0ffff000 ) |
        (( (1u + gas_idx) <<  0 ) & 0x00000fff )  
        ;
    return identity ; 

}

