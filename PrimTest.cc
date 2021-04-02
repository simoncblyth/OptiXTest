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
     pr.setAABB( extent ); 
     pr.setSbtIndexOffset(idx); 
     return pr ; 
}


/**
Highly inefficienct noddy appoach as not worth getting into 
thrust (or cudaMemcpy2D) complications for strided downloads just for this debug check 
**/

void DownloadDump( const PrimSpec& d_ps )
{
     for(unsigned i=0 ; i < d_ps.num_prim ; i++)
     { 
         const unsigned* u_ptr = d_ps.sbtIndexOffset + (d_ps.stride_in_bytes/sizeof(unsigned))*i ;  
         const float*    f_ptr = d_ps.aabb           + (d_ps.stride_in_bytes/sizeof(float))*i ;  

         float*     f = CU::DownloadArray<float>( f_ptr, 6 ); 
         unsigned*  u = CU::DownloadArray<unsigned>( u_ptr, 1 ); 

         std::cout << " off " << *(u) << " aabb (" << i << ") " ; 
         for( unsigned i=0 ; i < 6 ; i++ ) std::cout << *(f+i) << " " ; 
         std::cout << std::endl ; 

         delete [] f ; 
         delete [] u ;   
     }
}
 

void test_AABB()
{
    Prim p0 = {} ; 
    p0.setAABB( 42.42f ); 
    std::cout << "p0 " << p0.desc() << std::endl ; 

    Prim p1 = {} ;
    p1.setAABB( p0.AABB() ); 
    std::cout << "p1 " << p1.desc() << std::endl ; 
}



void test_offsets()
{
     std::cout << "test_offsets " << std::endl ; 
     std::cout 
        <<  "offsetof(struct Prim, q0) " <<  offsetof(struct Prim,  q0) << std::endl 
        <<  "offsetof(struct Prim, q0)/sizeof(float) " <<  offsetof(struct Prim, q0)/sizeof(float) << std::endl 
        ; 
     std::cout 
        <<  "offsetof(struct Prim, q1) " <<  offsetof(struct Prim,  q1) << std::endl 
        <<  "offsetof(struct Prim, q1)/sizeof(float) " <<  offsetof(struct Prim, q1)/sizeof(float) << std::endl 
        ; 

     std::cout 
        <<  "offsetof(struct Prim, q2) " <<  offsetof(struct Prim,  q2) << std::endl 
        <<  "offsetof(struct Prim, q2)/sizeof(float) " <<  offsetof(struct Prim, q2)/sizeof(float) << std::endl 
        ; 
     std::cout 
        <<  "offsetof(struct Prim, q3) " <<  offsetof(struct Prim,  q3) << std::endl 
        <<  "offsetof(struct Prim, q3)/sizeof(float) " <<  offsetof(struct Prim, q3)/sizeof(float) << std::endl 
        ; 
}


void test_spec( const std::vector<Prim>& prim )
{
     std::cout << "test_spec " << std::endl ; 
     PrimSpec psa = Prim::MakeSpec(prim.data(), 0, prim.size() ); 
     psa.dump(); 

     std::vector<float> out ; 
     psa.gather(out);  
     PrimSpec::Dump(out); 
}

void test_partial( const std::vector<Prim>& prim )
{
     std::cout << "test_partial " << std::endl ; 
     unsigned h = prim.size()/2 ; 

     PrimSpec ps0 = Prim::MakeSpec(prim.data(), 0, h ); 
     ps0.dump(); 

     PrimSpec ps1 = Prim::MakeSpec(prim.data(), h, h ); 
     ps1.dump(); 
}

Prim* test_upload( const Prim* prim, unsigned num )
{
     std::cout << "test_upload" << std::endl ; 
     Prim* d_prim = CU::UploadArray<Prim>(prim, num ) ;
     assert( d_prim ); 
     return d_prim ; 
}

void test_download( const Prim* d_prim, unsigned num )
{
     std::cout << "test_download" << std::endl ; 
     Prim* prim2 = CU::DownloadArray<Prim>( d_prim,  num ) ;
     for(unsigned i=0 ; i < num ; i++)
     {
         Prim* p = prim2 + i  ; 
         std::cout << i << std::endl << p->desc() << std::endl ; 
     }
}

PrimSpec test_dspec( Prim* d_prim , unsigned num)
{
     std::cout << "test_dspec" << std::endl ; 
     PrimSpec d_ps = Prim::MakeSpec( d_prim, 0, num ); 
     DownloadDump(d_ps);  

     return d_ps ; 
}



void test_pointer( const void* d, const char* label )
{
     std::cout << "test_pointer " << label << std::endl ; 

     const void* vd = (const void*) d ; 
     uintptr_t ud = (uintptr_t)d ; // uintptr_t is an unsigned integer type that is capable of storing a data pointer.
     CUdeviceptr cd = (CUdeviceptr) (uintptr_t) d ;  // CUdeviceptr is typedef to unsigned long lonh 

     std::cout << "            (const void*) d " << vd << std::endl ; 
     std::cout << "               (uintptr_t)d " << std::dec << ud << " " << std::hex << ud  << std::dec << std::endl ;  
     std::cout << "  (CUdeviceptr)(uintptr_t)d " << std::dec << cd << " " << std::hex << cd  << std::dec << std::endl ; 
}

int main(int argc, char** argv)
{
     test_AABB(); 
     test_offsets(); 

     std::vector<Prim> prim ; 
     for(unsigned i=0 ; i < 10 ; i++) prim.push_back(make_prim(float(i+1), i*10 )); 

     test_spec(prim); 
     test_partial(prim);
 
     Prim* d_prim = test_upload(prim.data(), prim.size());  
     test_download( d_prim, prim.size() ); 

     PrimSpec d_ps = test_dspec(d_prim, prim.size() ) ; 

     test_pointer( d_prim, "d_prim" ); 
     test_pointer( d_ps.aabb ,           "d_ps.aabb" ); 
     test_pointer( d_ps.sbtIndexOffset , "d_ps.sbtIndexOffset" ); 

     return 0 ; 
} 

