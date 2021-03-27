#pragma once
#include <stdint.h>
#include <vector_types.h>

struct Node ; 
struct Prim ; 

struct RaygenData
{
    float placeholder ; 
};

struct MissData
{
    float r, g, b;
};

struct HitGroupData
{
    Node*    node ;    // aka part 
    Prim*    prim ;    // (num_prim, 4 )          probably num_prim always 1 here  
    float4*  plan ;    // (num_plan,)             num_plan usually 0 
    //float4x4*   tran ;  // (num_tran, 3, 4, 4)     num_tran will eventually be same as num_node
};


#if defined(__CUDACC__) || defined(__CUDABE__)
#else
#include <optix_types.h>

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RaygenData>     Raygen ;
typedef SbtRecord<MissData>       Miss ;
typedef SbtRecord<HitGroupData>   HitGroup ;

#endif

