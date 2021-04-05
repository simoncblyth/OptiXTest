#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define HISTORY_FUNC __forceinline__ __device__
#else
#    define HISTORY_FUNC inline
#endif

/**
Using 64bit ull to store sequences of nibbles
**/

struct History
{
    enum {
           NUM = 2,     
           SIZE = 64,    // of each carrier     
           NITEM = 16,   // items within each 64bit
           NBITS = 4,     // bits per item    
           MASK  = 0xf 
         } ;  // 
    unsigned long long idx[NUM] ; 
    unsigned long long ctrl[NUM] ; 
    int curr ; 
};


HISTORY_FUNC
int history_append( History& h, unsigned idx, unsigned ctrl)
{
    if((h.curr+1) > h.NUM*h.NITEM ) return ERROR_OVERFLOW ; 
    h.curr++ ; 

    int nb = h.curr/h.NITEM  ;                             // target carrier int 
    unsigned long long  ii = h.curr*h.NBITS - h.SIZE*nb ; // bit offset within target 64bit 
    unsigned long long hidx = h.MASK & idx ;
    unsigned long long hctrl = h.MASK & ctrl ;

    h.idx[nb]  |=  hidx << ii   ; 
    h.ctrl[nb] |=  hctrl << ii  ; 

    return 0 ; 
}



