
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
#include "Solid.h"
#include "Node.h"
#include "Binding.h"
#include "Params.h"

#include "GAS.h"
#include "GAS_Builder.h"

#include "IAS.h"
#include "IAS_Builder.h"

#include "PIP.h"
#include "SBT.h"

#include "CU.h"


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
    geo(nullptr),
    is_1NN(false),
    is_11N(true)  
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


/**
SBT::setGeo
-------------

1. creates GAS using aabb obtained via geo
2. creates IAS
3. creates Hitgroup SBT records

**/

void SBT::setGeo(const Geo* geo_)
{
    geo = geo_ ; 

    createGAS(geo);      // uploads aabb of all prim of all shapes to create GAS
    createIAS(geo); 
    setTop(geo->top); 

    createHitgroup(geo); // creates Hitgroup SBT records   
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


/**
SBT::createGAS
----------------

For each compound shape the aabb of each prim (aka layer) is 
uploaded to GPU in order to create GAS for each compound shape.

Note that the prim could be a CSG tree of constituent nodes each 
with their own aabb, but only one aabb corresponding to the overall 
prim extent is used.


**/

void SBT::createGAS(const Geo* geo)
{
    unsigned num_solid = geo->getNumSolid(); 
    for(unsigned solidIdx=0 ; solidIdx < num_solid ; solidIdx++)
    {
        const Solid* so = geo->getSolid(solidIdx); 
        PrimSpec ps = geo->getPrimSpec(solidIdx); 

        GAS gas = {} ;  
        gas.so = so ; 
        GAS_Builder::Build(gas, ps);

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

/**
SBT::getOffset
----------------

The layer_idx_ within the shape_idx_ composite shape 
could also be called prim_idx_.



**/

unsigned SBT::getOffset(unsigned solid_idx_ , unsigned layer_idx_ ) const 
{
    assert( geo ); 
    unsigned num_gas = vgas.size(); 
    unsigned num_solid = geo->getNumSolid(); 
    assert( num_gas == num_solid ); 

    unsigned offset_sbt = _getOffset(solid_idx_, layer_idx_ ); 
 
    bool dump = false ; 
    if(dump) std::cout 
        << "SBT::getOffset"
        << " num_gas " <<  num_gas
        << " solid_idx_ " << solid_idx_
        << " layer_idx_ " << layer_idx_
        << " offset_sbt " << offset_sbt 
        << std::endl
        ;

    return offset_sbt ; 
}

/**
SBT::_getOffset
----------------

Implemented as an inner method avoiding "goto" 
to break out of multiple for loops.

**/
unsigned SBT::_getOffset(unsigned solid_idx_ , unsigned layer_idx_ ) const 
{
    unsigned offset_sbt = 0 ; 
    bool is_1NN = false ; 

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
                if( solid_idx_ == i && layer_idx_ == layer_idx ) return offset_sbt ;
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

/**
SBT::createHitgroup
---------------------

Note:

1. all HitGroup SBT records have the same hitgroup_pg, different shapes 
   are distinguished by program data not program code 


**/

void SBT::createHitgroup(const Geo* geo)
{
    unsigned num_solid = geo->getNumSolid(); 
    unsigned num_gas = vgas.size(); 
    assert( num_gas == num_solid ); 
    unsigned tot_rec = getTotalRec(); 

    std::cout 
        << "SBT::createHitgroup"
        << " num_solid " << num_solid 
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
        unsigned solid_idx = i ;    
        const GAS& gas = vgas[i] ;    
        unsigned num_bi = gas.bis.size(); 
        if(is_11N) assert( num_bi == 1 ); // 11N mode every GAS has only one BI with multiple aabb 

        const Solid* so = geo->getSolid(solid_idx) ;
        assert( gas.so == so );   

        int numPrim = so->numPrim ; 
        int primOffset = so->primOffset ; 

        std::cout << "SBT::createHitgroup gas_idx " << i << " num_bi " << num_bi << std::endl ; 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_rec = buildInputCPA.numSbtRecords ; 
            assert( num_rec == numPrim ) ; 

            if(is_1NN) assert( num_rec == 1 );  // 1NN mode : every BI has one aabb : THIS WAY CHOPS TO SMALLEST BBOX AND IS NO LONGER USED 

            for( unsigned k=0 ; k < num_rec ; k++)
            { 
                unsigned prim_idx = is_1NN ? j : k ;   
                unsigned check_sbt_offset = getOffset(solid_idx, prim_idx ); 
                bool expected_sbt_offset = check_sbt_offset == sbt_offset  ;


                unsigned globalPrimIdx = primOffset + prim_idx ;   
                const Prim* prim = geo->getPrim( globalPrimIdx ); 

                int numNode = prim->numNode(); 
                int nodeOffset = prim->nodeOffset();  

                std::cout 
                    << "SBT::createHitgroup "
                    << " gas(i) " << i 
                    << " bi(j) " << j
                    << " sbt(k) " << k 
                    << " solid_idx " << solid_idx 
                    << " prim_idx " << prim_idx 
                    << " check_sbt_offset " << check_sbt_offset
                    << " sbt_offset " << sbt_offset
                    << " numNode " << numNode
                    << " nodeOffset " << nodeOffset
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

                hg->data.numNode = numNode ; 
                hg->data.nodeOffset = nodeOffset ; 
  
                //upload_prim_data( hg->data, sh, prim_idx );                 

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

Sets device pointers into HitGroupData *data* CPU struct which is about to be copied to device


void SBT::upload_prim_data( HitGroupData& data, const Shape* sh, unsigned prim_idx )
{
    unsigned num_prim = 1 ;  // expect this to always be 1 : single prim (aka layer) for each HG SBT record
   
    int* prim = sh->get_prim(prim_idx) ; 
    int* d_prim = CU::UploadArray<int>(prim, Shape::prim_size*num_prim ) ; 
    data.prim = (Prim*)d_prim ; 

    unsigned num_node = sh->get_num_node( prim_idx ); 
    assert( num_node == 1 ) ; // only single node "trees" for now 
    
    const Node* node = sh->get_node(prim_idx); 
    Node* d_node = CU::UploadArray<Node>(node, num_node ); 
    data.node = d_node ; 

}

void SBT::check_prim_data( const HitGroupData& data ) const 
{
    std::cout << "SBT::check_prim_data" << std::endl ; 

    unsigned num_prim = 1 ;  // expect this to always be 1  
    const int* d_prim = (const int*)data.prim ; 
    int* prim = CU::DownloadArray<int>(d_prim, Shape::prim_size*num_prim );  
    int num_node = prim[1] ; 

    std::cout << " prim " ; 
    for(int i=0 ; i < 4 ; i++) std::cout << prim[i] << " " ; 
    std::cout << std::endl ;  

    std::cout << " num_node " << num_node << std::endl ; 
    Node* d_node = data.node ; 
    Node* node = CU::DownloadArray<Node>(d_node, num_node);  

    Node::Dump( node, 1, "SBT::check_prim_data"); 
}

**/



void SBT::checkHitgroup(const Geo* geo)
{
    unsigned tot_sbt = sbt.hitgroupRecordCount ;
    std::cout 
        << "SBT::checkHitgroup" 
        << " tot_sbt " << tot_sbt
        << std::endl 
        ; 

    check = new HitGroup[tot_sbt] ; 

    CUDA_CHECK( cudaMemcpy(check, reinterpret_cast<void*>( sbt.hitgroupRecordBase ), sizeof( HitGroup )*tot_sbt, cudaMemcpyDeviceToHost ));
    HitGroup* hg = check ; 
    for(unsigned i=0 ; i < tot_sbt ; i++)
    {
        //check_prim_data( hg->data ); 
        hg++ ; 
    }
}



