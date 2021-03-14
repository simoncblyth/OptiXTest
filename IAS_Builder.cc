#include <iostream>
#include <iomanip>
#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp

#include "CUDA_CHECK.h"
#include "OPTIX_CHECK.h"

#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Grid.h"
#include "Ctx.h"

#include "InstanceId.h"
#include "GAS.h"
#include "IAS.h"
#include "IAS_Builder.h"
#include "SBT.h"

/**
IAS_Builder::Build
--------------------

Converts a grid with geometry identity instrumented transforms into
a vector of OptixInstance. The instance.sbtOffset are set using SBT::getOffset
for the gas_idx and with prim_idx:0 indicating the outer prim(aka layer) 
of the GAS.

**/

void IAS_Builder::Build(IAS& ias, const Grid* gr, const SBT* sbt) // static 
{
    unsigned num_tr = gr->trs.size() ; 
    std::cout << "IAS_Builder::Build num_tr " << num_tr << std::endl ; 
    assert( num_tr > 0); 
    const float* vals =   (float*)gr->trs.data() ;
    unsigned flags = OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT ;  
 
    std::vector<OptixInstance> instances ;  
    for(unsigned i=0 ; i < num_tr ; i++)
    {
        glm::mat4 mat(1.0f) ; 
        memcpy( glm::value_ptr(mat), (void*)(vals + i*16), 16*sizeof(float));
        
        glm::mat4 imat = glm::transpose(mat);

        glm::uvec4 idv ; // after transposiing the last row contains the identity info 
        memcpy( glm::value_ptr(idv), &imat[3], 4*sizeof(float) ); 


        unsigned identity = idv.x ; 
        unsigned ins_idx ; 
        unsigned gas_idx ; 
        InstanceId::Decode( ins_idx, gas_idx, identity );
        unsigned prim_idx = 0u ;  // need offset for the outer prim(aka layer) of the GAS 

        const GAS& gas = sbt->getGAS(gas_idx); 

        OptixInstance instance = {} ; 
        memcpy( instance.transform, glm::value_ptr(imat), 12*sizeof( float ) );
        instance.instanceId = identity ; 
        instance.sbtOffset = sbt->getOffset(gas_idx, prim_idx );            
        instance.visibilityMask = 255;
        instance.flags = flags ;
        instance.traversableHandle = gas.handle ; 
    
        instances.push_back(instance); 
    }
    Build(ias, instances); 
}

/**
IAS_Builder::Build
-------------------

Boilerplate turning the vector of OptixInstance into an IAS.

**/

void IAS_Builder::Build(IAS& ias, const std::vector<OptixInstance>& instances)
{
    unsigned numInstances = instances.size() ; 
    std::cout << "IAS_Builder::bBild numInstances " << numInstances << std::endl ; 

    unsigned numBytes = sizeof( OptixInstance )*numInstances ; 


    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ias.d_instances ), numBytes ) );
    CUDA_CHECK( cudaMemcpy(
                reinterpret_cast<void*>( ias.d_instances ),
                instances.data(),
                numBytes,
                cudaMemcpyHostToDevice
                ) );

 
    OptixBuildInput buildInput = {};

    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = ias.d_instances ; 
    buildInput.instanceArray.numInstances = numInstances ; 


    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = 
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION ;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes as_buffer_sizes;

    OPTIX_CHECK( optixAccelComputeMemoryUsage( Ctx::context, &accel_options, &buildInput, 1, &as_buffer_sizes ) );
    CUdeviceptr d_temp_buffer_as;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_temp_buffer_as ), as_buffer_sizes.tempSizeInBytes ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_as_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( as_buffer_sizes.outputSizeInBytes, 8ull );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_as_and_compacted_size ),
                compactedSizeOffset + 8
                ) );

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_as_and_compacted_size + compactedSizeOffset );


    OPTIX_CHECK( optixAccelBuild( Ctx::context,
                                  0,                  // CUDA stream
                                  &accel_options,
                                  &buildInput,
                                  1,                  // num build inputs
                                  d_temp_buffer_as,
                                  as_buffer_sizes.tempSizeInBytes,
                                  d_buffer_temp_output_as_and_compacted_size,
                                  as_buffer_sizes.outputSizeInBytes,
                                  &ias.handle,
                                  &emitProperty,      // emitted property list
                                  1                   // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_as ) );

    size_t compacted_as_size;
    CUDA_CHECK( cudaMemcpy( &compacted_as_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );


    if( compacted_as_size < as_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &ias.d_buffer ), compacted_as_size ) );

        // use ias.handle as input and output
        OPTIX_CHECK( optixAccelCompact( Ctx::context, 0, ias.handle, ias.d_buffer, compacted_as_size, &ias.handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_as_and_compacted_size ) );

        std::cout 
            << "IAS::build (compacted is smaller) "
            << " compacted_as_size : " << compacted_as_size
            << " as_buffer_sizes.outputSizeInBytes : " << as_buffer_sizes.outputSizeInBytes
            << std::endl
            ; 

    }
    else
    {
        ias.d_buffer = d_buffer_temp_output_as_and_compacted_size;

        std::cout 
            << "IAS_Builder::Build (compacted not smaller) "
            << " compacted_as_size : " << compacted_as_size
            << " as_buffer_sizes.outputSizeInBytes : " << as_buffer_sizes.outputSizeInBytes
            << std::endl
            ; 
    }
}


