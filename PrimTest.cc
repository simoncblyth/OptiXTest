#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include "cuda.h"

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




void DownloadDump( const PrimSpec& d_ps )
{
     for(unsigned i=0 ; i < d_ps.num_prim ; i++)
     { 
         unsigned* u_ptr = d_ps.sbtIndexOffset + (d_ps.stride_in_bytes/sizeof(unsigned))*i ;  
         float*    f_ptr = d_ps.aabb           + (d_ps.stride_in_bytes/sizeof(float))*i ;  

         // highly inefficienct noddy appoach as not woth getting into 
         // thrust complications for strided downloads just for this debug check 

         float*     f = CU::DownloadArray<float>( f_ptr, 6 ); 
         unsigned*  u = CU::DownloadArray<unsigned>( u_ptr, 1 ); 

         std::cout << " off " << *(u) << " aabb (" << i << ") " ; 
         for( unsigned i=0 ; i < 6 ; i++ ) std::cout << *(f+i) << " " ; 
         std::cout << std::endl ; 

         delete [] f ; 
         delete [] u ;   
     }
}
 

int main(int argc, char** argv)
{
     unsigned num = 10 ; 

     std::vector<Prim> prim ; 
     for(unsigned i=0 ; i < num ; i++) prim.push_back(make_prim(float(i+1), i*10 )); 

     PrimSpec psa = Prim::MakeSpec(prim.data(), 0, prim.size() ); 
     psa.dump(); 


     std::vector<float> out ; 
     psa.gather(out);  
     PrimSpec::Dump(out); 

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

/*
     Prim* prim2 = CU::DownloadArray<Prim>( d_prim,  num ) ;
     for(unsigned i=0 ; i < num ; i++)
     {
         Prim* p = prim2 + i  ; 
         std::cout << i << std::endl << p->desc() << std::endl ; 
     }
*/

     PrimSpec d_psd = Prim::MakeSpec( d_prim, 0, num ); 
     DownloadDump(d_psd);  



     CUdeviceptr dptr = (CUdeviceptr) (uintptr_t) d_psd.aabb ;

     // uintptr_t is an unsigned integer type that is capable of storing a data pointer.
     // CUdeviceptr is typedef to unsigned long lonh 
     std::cout << "                          d_psd.aabb " <<                         d_psd.aabb << std::endl ;  
     std::cout << "               (uintptr_t)d_psd.aabb " <<              (uintptr_t)d_psd.aabb << std::endl ;  
     std::cout << "  (CUdeviceptr)(uintptr_t)d_psd.aabb " << (CUdeviceptr)(uintptr_t)d_psd.aabb << std::endl ; 

     return 0 ; 
} 

