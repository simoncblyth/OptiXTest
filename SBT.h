#pragma once

#include <optix.h>
#include <vector>
#include "Binding.h"
#include "GAS.h"
#include "IAS.h"

/**
SBT : RG,MS,HG program data preparation 
===========================================

Aim to minimize geometry specifics in here ...


**/
struct PIP ; 
struct Geo ; 

struct SBT 
{
    const PIP*    pip ; 
    Raygen*       raygen ;
    Miss*         miss ;
    HitGroup*     hitgroup ;
    HitGroup*     check ;
    const Geo*    geo ; 
    bool          is_1NN ;  // 1NN:true is smallest bbox chopped 
    bool          is_11N ; 
 
    CUdeviceptr   d_raygen ;
    CUdeviceptr   d_miss ;
    CUdeviceptr   d_hitgroup ;

    OptixShaderBindingTable sbt = {};

    std::vector<GAS> vgas ; 
    std::vector<IAS> vias ; 
    AS*              top ; 


    SBT( const PIP* pip_ ); 
    void setGeo(const Geo* geo_); 

    AS* getAS(const char* spec) const ;
    void setTop(const char* spec) ;
    void setTop(AS* top_) ;
    AS* getTop() const ;


    void init();  
    void createRaygen();  
    void updateRaygen();  

    void createMiss();  
    void updateMiss();  

    void createGAS(const Geo* geo);
    void createIAS(const Geo* geo);

    const GAS& getGAS(unsigned gas_idx) const ;
    const IAS& getIAS(unsigned ias_idx) const ;

    unsigned getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ; 
    unsigned _getOffset(unsigned shape_idx_ , unsigned layer_idx_ ) const ;

    unsigned getTotalRec() const ;
    void createHitgroup(const Geo* geo);

    void checkHitgroup(const Geo* geo); 

    //void check_prim_data( const HitGroupData& data ) const ;
    //void upload_prim_data( HitGroupData& data, const Shape* sh, unsigned prim_idx );

};

