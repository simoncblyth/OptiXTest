
#include <iostream>
#include <iomanip>

#include "Ctx.h"
#include "AS.h"
#include "Params.h"

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK

OptixDeviceContext Ctx::context = nullptr ;

void Ctx::context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)  // static 
{
    std::cerr 
        << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
        << message << "\n";
}

Ctx::Ctx(Params* params_)
    :
    params(params_)
{
    CUDA_CHECK( cudaFree( 0 ) ); 

    CUcontext cuCtx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &Ctx::context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
}

void Ctx::setTop(const AS* top)
{
    params->handle = top->handle ; 
}

void Ctx::uploadParams()
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( d_param ),
                params, sizeof( Params ),
                cudaMemcpyHostToDevice
                ) );
}


