#pragma once

#include <glm/glm.hpp>
#include <optix.h>

struct Params ; 
struct AS ; 

struct Ctx 
{
    Params*     params ; 
    CUdeviceptr d_param;
    static OptixDeviceContext context ;

    static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */); 

    Ctx(Params* params_); 

    void setTop(const AS* top);

    void uploadParams();

};

