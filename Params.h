#pragma once

#include <optix.h>
#include <vector_types.h>

#ifndef __CUDACC__
#include <glm/glm.hpp>
#endif

struct Node ; 
struct qat4 ; 

struct Params
{
    Node*      node ; 
    float4*    plan ; 
    qat4*      tran ; 
    qat4*      itra ; 

    uchar4*    pixels ;
    float4*    isect ;

    uint32_t   width;
    uint32_t   height;
    uint32_t   depth;
    uint32_t   cameratype ; 

    int32_t    origin_x;
    int32_t    origin_y;

    float3     eye;
    float3     U ;
    float3     V ; 
    float3     W ;
    float      tmin ; 
    float      tmax ; 

#if OPTIX_VERSION < 70000
    void*                   handle ;  
#else
    OptixTraversableHandle  handle ; 
#endif

#ifndef __CUDACC__
    void setView(const glm::vec3& eye_, const glm::vec3& U_, const glm::vec3& V_, const glm::vec3& W_, float tmin_, float tmax_, unsigned cameratype );
    void setSize(unsigned width_, unsigned height_, unsigned depth_ );
#endif


};

