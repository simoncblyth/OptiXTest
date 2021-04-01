#include <vector>
#include <iostream>
#include <cassert>

#include "sutil_vec_math.h"
#include "Prim.h"
#include "CU.h"

Prim make_prim( float extent, unsigned idx )
{
     Prim pr = {} ; 
     pr.mn.x = -extent ; 
     pr.mn.y = -extent ; 
     pr.mn.z = -extent ; 
     pr.mx.x =  extent ; 
     pr.mx.y =  extent ; 
     pr.mx.z =  extent; 
     pr.sbtIndexOffset = idx ; 
     return pr ; 
}

int main(int argc, char** argv)
{
     unsigned num = 10 ; 

     std::vector<Prim> prim ; 
     for(unsigned i=0 ; i < num ; i++) prim.push_back(make_prim(float(i+1), i*10 )); 

     PrimSpec psa = Prim::MakeSpec(prim.data(), 0, prim.size() ); 
     psa.dump(); 


     unsigned h = num/2 ; 

     PrimSpec ps0 = Prim::MakeSpec(prim.data(), 0, h ); 
     ps0.dump(); 

     PrimSpec ps1 = Prim::MakeSpec(prim.data(), h, h ); 
     ps1.dump(); 


     std::cout 
        <<  "offsetof(struct Prim, sbtIndexOffset) " 
        <<  offsetof(struct Prim, sbtIndexOffset) 
        << std::endl 
        <<  "offsetof(struct Prim, sbtIndexOffset)/sizeof(float) " 
        <<  offsetof(struct Prim, sbtIndexOffset)/sizeof(float) 
        << std::endl 
        ; 


     Prim* d_prim = CU::UploadArray<Prim>(prim.data(), num ) ;
     assert( d_prim ); 


     PrimSpec d_psd = Prim::MakeSpec( d_prim, 0, num ); 
      
     unsigned* offsets = CU::DownloadArray<unsigned>( d_psd.sbtIndexOffset, num ) ; 

     std::cout << "d_psd sbtIndexOffset " ; 
     for( unsigned i=0 ; i < num ; i++ ) std::cout << *(offsets+i) << " "  ; 
     std::cout << std::endl ; 


     return 0 ; 
} 

