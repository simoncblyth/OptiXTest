
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
    //glm::vec3 purple(0.3f, 0.1f, 0.5f); 
    //glm::vec3 white( 1.0f, 1.0f, 1.0f); 
    glm::vec3 lightgrey( 0.9f, 0.9f, 0.9f); 
    const glm::vec3& bkg = lightgrey  ; 
   
    miss->data.r = bkg.x ;
    miss->data.g = bkg.y ;
    miss->data.b = bkg.z ;

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

void SBT::createGAS(const Geo* geo)   // just pass-thru to Foundry 
{
    unsigned num_solid = geo->getNumSolid(); 
    for(unsigned solidIdx=0 ; solidIdx < num_solid ; solidIdx++)
    {
        PrimSpec ps = geo->getPrimSpec(solidIdx); 
        GAS gas = {} ;  
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

The layer_idx_ within the shape_idx_ composite shape.
NB layer_idx is local to the solid. 

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
        unsigned solidIdx = i ;    
        const GAS& gas = vgas[i] ;    
        unsigned num_bi = gas.bis.size(); 
        if(is_11N) assert( num_bi == 1 ); // 11N mode every GAS has only one BI with aabb for each Prim 

        const Solid* so = geo->getSolid(solidIdx) ;
        int numPrim = so->numPrim ; 
        int primOffset = so->primOffset ; 

        std::cout << "SBT::createHitgroup solidIdx " << solidIdx << " so.numPrim " << numPrim << " so.primOffset " << primOffset << std::endl ; 

        for(unsigned j=0 ; j < num_bi ; j++)
        { 
            const BI& bi = gas.bis[j] ; 
            const OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;
            unsigned num_rec = buildInputCPA.numSbtRecords ; 
            assert( num_rec == numPrim ) ; 

            if(is_1NN) assert( num_rec == 1 );  // 1NN mode : every BI has one aabb : THIS WAY CHOPS TO SMALLEST BBOX AND IS NO LONGER USED 

            for( unsigned k=0 ; k < num_rec ; k++)
            { 
                unsigned localPrimIdx = is_1NN ? j : k ;   
                unsigned globalPrimIdx = primOffset + localPrimIdx ;   
                const Prim* prim = geo->getPrim( globalPrimIdx ); 
                setPrimData( hg->data, prim ); 
                dumpPrimData( hg->data ); 

                unsigned check_sbt_offset = getOffset(solidIdx, localPrimIdx ); 
                std::cout 
                    << "SBT::createHitgroup "
                    << " gas(i) " << i 
                    << " bi(j) " << j
                    << " sbt(k) " << k 
                    << " solidIdx " << solidIdx 
                    << " localPrimIdx " << localPrimIdx 
                    << " globalPrimIdx " << globalPrimIdx 
                    << " check_sbt_offset " << check_sbt_offset
                    << " sbt_offset " << sbt_offset
                    << std::endl 
                    ; 
                assert( check_sbt_offset == sbt_offset  ); 

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

void SBT::setPrimData( HitGroupData& data, const Prim* prim)
{
    data.numNode = prim->numNode(); 
    data.nodeOffset = prim->nodeOffset();  
    //data.tranOffset = prim->tranOffset();  
    //data.planOffset = prim->planOffset();  
}

void SBT::checkPrimData( HitGroupData& data, const Prim* prim)
{
    assert( data.numNode == prim->numNode() ); 
    assert( data.nodeOffset == prim->nodeOffset() );  
    //assert( data.tranOffset == prim->tranOffset() );  
    //assert( data.planOffset == prim->planOffset() );  
}
void SBT::dumpPrimData( const HitGroupData& data ) const 
{
    std::cout 
        << "SBT::dumpPrimData"
        << " data.numNode " << data.numNode
        << " data.nodeOffset " << data.nodeOffset
      //  << " data.tranOffset " << data.tranOffset
      //  << " data.planOffset " << data.planOffset
        << std::endl 
        ; 
}

void SBT::checkHitgroup(const Geo* geo)
{
    unsigned num_solid = geo->getNumSolid(); 
    unsigned num_prim = geo->getNumPrim(); 
    unsigned num_sbt = sbt.hitgroupRecordCount ;
    std::cout 
        << "SBT::checkHitgroup" 
        << " num_sbt " << num_sbt
        << " num_solid " << num_solid
        << " num_prim " << num_prim
        << std::endl 
        ; 

    assert( num_prim == num_sbt ); 

    check = new HitGroup[num_prim] ; 
    CUDA_CHECK( cudaMemcpy(check, reinterpret_cast<void*>( sbt.hitgroupRecordBase ), sizeof( HitGroup )*num_prim, cudaMemcpyDeviceToHost ));
    HitGroup* hg = check ; 
    for(unsigned i=0 ; i < num_prim ; i++)
    {
        unsigned globalPrimIdx = i ; 
        const Prim* prim = geo->getPrim(globalPrimIdx);         

        dumpPrimData( hg->data ); 
        checkPrimData( hg->data, prim ); 

        hg++ ; 
    }

    delete [] check ; 
}

