#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <fstream>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp
#include "OPTIX_CHECK.h"
#include "CUDA_CHECK.h"

#include <glm/glm.hpp>


#include "Geo.h"
#include "Shape.h"
#include "Node.h"
#include "Binding.h"
#include "Params.h"

#include "GAS.h"
#include "GAS_Builder.h"

#include "IAS.h"
#include "IAS_Builder.h"

#include "PIP.h"
#include "SBT.h"

/**
SBT
====

SBT needs PIP as the packing of SBT record headers requires 
access to their corresponding program groups (PGs).  
This is one aspect of establishing the connection between the 
PGs and their data.

**/

SBT::SBT(const PIP* pip_)
    :
    pip(pip_),
    raygen(nullptr),
    miss(nullptr),
    hitgroup(nullptr),
    check(nullptr),
    geo(nullptr)
{
    init(); 
}

void SBT::init()
{
    std::cout << "SBT::init" << std::endl ; 
    createRaygen();
    updateRaygen();

    createMiss();
    updateMiss(); 
}

void SBT::setGeo(const Geo* geo_)
{
    geo = geo_ ; 

    createGAS(geo); 
    createIAS(geo); 
    setTop(geo->top); 

    createHitgroup(geo); 
    checkHitgroup(geo); 
}


/**
SBT::createMissSBT
--------------------

NB the records have opaque header and user data
**/

void SBT::createMiss()
{
    miss = new Miss ; 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_miss ), sizeof(Miss) ) );
    sbt.missRecordBase = d_miss;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->miss_pg, miss ) );

    sbt.missRecordStrideInBytes     = sizeof( Miss );
    sbt.missRecordCount             = 1;
}

void SBT::updateMiss()
{
    miss->data.r = 0.3f ;
    miss->data.g = 0.1f ;
    miss->data.b = 0.5f ;

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_miss ),
                miss,
                sizeof(Miss),
                cudaMemcpyHostToDevice
                ) );
}

void SBT::createRaygen()
{
    raygen = new Raygen ; 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_raygen ),   sizeof(Raygen) ) );
    sbt.raygenRecord = d_raygen;
    OPTIX_CHECK( optixSbtRecordPackHeader( pip->raygen_pg,   raygen ) );
}

void SBT::updateRaygen()
{
    std::cout <<  "SBT::updateRaygen " << std::endl ; 

    raygen->data = {};
    raygen->data.placeholder = 42.0f ;

    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_raygen ),
                raygen,
                sizeof( Raygen ),
                cudaMemcpyHostToDevice
                ) );
}


void SBT::createGAS(const Geo* geo)
{
    unsigned num_shape = geo->getNumShape(); 
    for(unsigned i=0 ; i < num_shape ; i++)
    {
        const Shape* sh = geo->getShape(i) ;    
        GAS gas = {} ;  
        GAS_Builder::Build(gas, sh );
        vgas.push_back(gas);  
    }
}

void SBT::createIAS(const Geo* geo)
{
    unsigned num_grid = geo->getNumGrid(); 
    for(unsigned i=0 ; i < num_grid ; i++)
    {
        const Grid* gr = geo->getGrid(i) ;    
        IAS ias = {} ;  
        IAS_Builder::Build(ias, gr, this );
        vias.push_back(ias);  
    }
}

const GAS& SBT::getGAS(unsigned gas_idx) const 
{
    assert( gas_idx < vgas.size()); 
    return vgas[gas_idx]; 
}

const IAS& SBT::getIAS(unsigned ias_idx) const 
{
    assert( ias_idx < vias.size()); 
    return vias[ias_idx]; 
}


AS* SBT::getTop() const 
{
    return top ; 
}

void SBT::setTop(const char* spec)
{
    AS* a = getAS(spec); 
    setTop(a); 
}
void SBT::setTop(AS* top_)
{   
    top = top_ ;
}

AS* SBT::getAS(const char* spec) const 
{
   assert( strlen(spec) > 1 );  
   char c = spec[0]; 
   assert( c == 'i' || c == 'g' );  
   int idx = atoi( spec + 1 );  

   std::cout << "SBT::getAS " << spec << " c " << c << " idx " << idx << std::endl ; 

   AS* a = nullptr ; 
   if( c == 'i' )
   {   
       const IAS& ias = vias[idx]; 
       a = (AS*)&ias ; 
   }   
   else if( c == 'g' )
   {   
       const GAS& gas = vgas[idx]; 
       a = (AS*)&gas ; 
   }   

   if(a)
   {   
       std::cout << "SBT::getAS " << spec << std::endl ; 
   }   
   return a ; 
}

unsigned SBT::getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const 
{
    assert( geo ); 
    bool is_1NN = geo->gas_bi_aabb == 0u ;  
    bool is_11N = geo->gas_bi_aabb == 1u ; 
    assert( is_1NN || is_11N  ); 

    unsigned num_gas = vgas.size(); 
    unsigned num_shape = geo->getNumShape(); 
    assert( num_gas == num_shape ); 

    unsigned offset_sbt = _getOffset(shape_idx_, layer_idx_ ); 
 
    bool dump = false ; 
    if(dump) std::cout 
        << "SBT::getOffset"
        << " num_gas " <<  num_gas
        << " shape_idx_ " << shape_idx_
        << " layer_idx_ " << layer_idx_
        << " offset_sbt " << offset_sbt 
        << std::endl
        ;

    return offset_sbt ; 
}

unsigned SBT::_getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const 
{
    unsigned offset_sbt = 0 ; 
    bool is_1NN = geo->gas_bi_aabb == 0u ;  
    for(unsigned i=0 ; i < vgas.size() ; i++)
    {
        const GAS& gas = vgas[i] ;    
        unsigned num_bi = gas.bis.size(); 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_sbt = buildInputCPA.numSbtRecords ; 
            if(is_1NN) assert( num_sbt == 1 );  

            for( unsigned k=0 ; k < num_sbt ; k++)
            { 
                unsigned layer_idx = is_1NN ? j : k ;  
                if( shape_idx_ == i && layer_idx_ == layer_idx ) return offset_sbt ;
                offset_sbt += 1 ; 
            }
        }         
    }
    std::cout << "SBT::_getOffsetSBT WARNING : did not find targetted shape " << std::endl ; 
    assert(0); 
    return offset_sbt ;  
}


unsigned SBT::getTotalRec() const 
{
    bool is_1NN = geo->gas_bi_aabb == 0u ;  
    unsigned tot_bi = 0 ; 
    unsigned tot_rec = 0 ; 
    for(unsigned i=0 ; i < vgas.size() ; i++)
    {
        const GAS& gas = vgas[i] ;    
        unsigned num_bi = gas.bis.size(); 
        tot_bi += num_bi ; 
        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_rec = buildInputCPA.numSbtRecords ; 
            if(is_1NN) assert( num_rec == 1 );  
            tot_rec += num_rec ; 
        }         
    }
    assert( tot_bi > 0 && tot_rec > 0 );  
    if(is_1NN) assert( tot_bi == tot_rec );  
    return tot_rec ;  
}


void SBT::createHitgroup(const Geo* geo)
{
    bool is_1NN = geo->gas_bi_aabb == 0u ;  
    bool is_11N = geo->gas_bi_aabb == 1u ; 
    assert( is_1NN || is_11N  ); 

    unsigned num_shape = geo->getNumShape(); 
    unsigned num_gas = vgas.size(); 
    assert( num_gas == num_shape ); 
    unsigned tot_rec = getTotalRec(); 

    std::cout 
        << "SBT::createHitgroup"
        << " num_shape " << num_shape 
        << " num_gas " << num_gas 
        << " tot_rec " << tot_rec 
        << std::endl 
        ; 

    hitgroup = new HitGroup[tot_rec] ; 
    HitGroup* hg = hitgroup ; 

    for(unsigned i=0 ; i < tot_rec ; i++)   // pack headers CPU side
         OPTIX_CHECK( optixSbtRecordPackHeader( pip->hitgroup_pg, hitgroup + i ) ); 
    
    unsigned sbt_offset = 0 ; 
    for(unsigned i=0 ; i < num_gas ; i++)
    {
        unsigned shape_idx = i ;    
        const GAS& gas = vgas[i] ;    
        unsigned num_bi = gas.bis.size(); 
        const Shape* sh = gas.sh ; 

        std::cout << "SBT::createHitgroup gas_idx " << i << " num_bi " << num_bi << std::endl ; 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_rec = buildInputCPA.numSbtRecords ; 
            if(is_1NN) assert( num_rec == 1 ); 

            for( unsigned k=0 ; k < num_rec ; k++)
            { 
                unsigned prim_idx = is_1NN ? j : k ;   
                unsigned check_sbt_offset = getOffset(shape_idx, prim_idx ); 
                bool expected_sbt_offset = check_sbt_offset == sbt_offset  ;

                std::cout 
                    << "SBT::createHitgroup "
                    << " gas(i) " << i 
                    << " bi(j) " << j
                    << " sbt(k) " << k 
                    << " shape_idx " << shape_idx 
                    << " prim_idx " << prim_idx 
                    << " check_sbt_offset " << check_sbt_offset
                    << " sbt_offset " << sbt_offset
                    << std::endl 
                    ; 

                if(!expected_sbt_offset) 
                   std::cout 
                      << "SBT::createHitgroup FATAL "
                      << " sbt_offset " << sbt_offset 
                      << " check_sbt_offset " << check_sbt_offset 
                      << std::endl
                      ;
                assert( expected_sbt_offset ); 

                upload_prim_data( hg->data, sh, prim_idx );                 

                hg++ ; 
                sbt_offset++ ; 
            }
        }
    }

    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_hitgroup ), sizeof(HitGroup)*tot_rec ));
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_hitgroup ), hitgroup, sizeof(HitGroup)*tot_rec, cudaMemcpyHostToDevice ));

    sbt.hitgroupRecordBase  = d_hitgroup;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroup);
    sbt.hitgroupRecordCount = tot_rec ;
}


/**
SBT::upload_prim_data
------------------------

Sets device pointers into *data* CPU struct which is about to be copied to device

**/

void SBT::upload_prim_data( HitGroupData& data, const Shape* sh, unsigned prim_idx )
{
    unsigned num_prim = 1 ;  // expect this to always be 1  
    int* prim = sh->get_prim(prim_idx) ; 
    int* d_prim = UploadArray<int>(prim, Shape::prim_size*num_prim ) ; 
    data.prim = d_prim ; 

    unsigned num_node = sh->get_num_node( prim_idx ); 
    assert( num_node == 1 ) ; // only single node "trees" for now 
    
    const Node* node = sh->get_node(prim_idx); 
    Node* d_node = UploadArray<Node>(node, num_node ); 
    data.node = d_node ; 
}

void SBT::check_prim_data( const HitGroupData& data ) const 
{
    std::cout << "SBT::check_prim_data" << std::endl ; 

    unsigned num_prim = 1 ;  // expect this to always be 1  
    int* d_prim = data.prim ; 
    int* prim = DownloadArray<int>(d_prim, Shape::prim_size*num_prim );  
    int num_node = prim[1] ; 

    std::cout << " prim " ; 
    for(int i=0 ; i < 4 ; i++) std::cout << prim[i] << " " ; 
    std::cout << std::endl ;  

    std::cout << " num_node " << num_node << std::endl ; 
    Node* d_node = data.node ; 
    Node* node = DownloadArray<Node>(d_node, num_node);  

    Node::Dump( node, 1, "SBT::check_prim_data"); 
}


void SBT::checkHitgroup(const Geo* geo)
{
    unsigned tot_sbt = sbt.hitgroupRecordCount ;
    std::cout 
        << "SBT::checkHitgroup" 
        << " tot_sbt " << tot_sbt
        << std::endl 
        ; 
    assert( geo->gas_bi_aabb == 0u || geo->gas_bi_aabb == 1u ); 

    check = new HitGroup[tot_sbt] ; 

    CUDA_CHECK( cudaMemcpy(check, reinterpret_cast<void*>( sbt.hitgroupRecordBase ), sizeof( HitGroup )*tot_sbt, cudaMemcpyDeviceToHost ));
    HitGroup* hg = check ; 
    for(unsigned i=0 ; i < tot_sbt ; i++)
    {
        check_prim_data( hg->data ); 
        hg++ ; 
    }
}


/**
SBT::UploadArray
----------------

Allocate on device and copy from host to device

**/
template <typename T>
T* SBT::UploadArray(const T* array, unsigned num_items ) // static
{
    std::cout << "SBT::UploadArray num_items " << num_items << std::endl ; 
    T* d_array = nullptr ; 
    CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>( &d_array ), num_items*sizeof(T) ));
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>( d_array ), array, sizeof(T)*num_items, cudaMemcpyHostToDevice ));
    return d_array ; 
}

/**
SBT::UploadArray  
----------------

Allocate on host and copy from device to host 

**/

template <typename T>
T* SBT::DownloadArray(const T* d_array, unsigned num_items ) // static
{
    std::cout << "SBT::DownloadArray num_items " << num_items << std::endl ; 
    T* array = new T[num_items] ;  
    CUDA_CHECK( cudaMemcpy( array, d_array, sizeof(T)*num_items, cudaMemcpyDeviceToHost ));
    return array ; 
}

template float* SBT::UploadArray<float>(const float* array, unsigned num_items) ;
template float* SBT::DownloadArray<float>(const float* d_array, unsigned num_items) ;

template Node* SBT::UploadArray<Node>(const Node* d_array, unsigned num_items) ;
template Node* SBT::DownloadArray<Node>(const Node* d_array, unsigned num_items) ;

