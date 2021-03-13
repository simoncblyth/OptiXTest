
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_vec_math.h"    // roundUp

#include "CUDA_CHECK.h"
#include "OPTIX_CHECK.h"

#include "Ctx.h"
#include "Shape.h"
#include "GAS.h"
#include "GAS_Builder.h"

/**
GAS_Builder::Build
-------------------

**/

void GAS_Builder::Build(GAS& gas, const Shape* sh )  // static
{
    if(sh->is1NN())
    {
        Build_1NN(gas, sh);  
    }
    else if(sh->is11N())
    {
        Build_11N(gas, sh);  
    }
    else
    {
        assert(0); 
    }
}

/**
GAS_Builder::Build_1NN GAS:BI:AABB  1:N:N  with a single AABB for each BI
-----------------------------------------------------------------------------

This way gets bbox chopped with 700, Driver 435.21

**/

void GAS_Builder::Build_1NN( GAS& gas, const Shape* sh )
{
    std::cout 
        << "GAS_Builder::Build_1NN"
        << " sh.num " << sh->num 
        << " sh.kludge_outer_aabb " << sh->kludge_outer_aabb
        << std::endl
        ;  
    gas.sh = sh ; 

    for(unsigned i=0 ; i < sh->num ; i++)
    { 
         BI bi = MakeCustomPrimitivesBI_1NN( sh,  i );  
         gas.bis.push_back(bi); 
    }
    std::cout << "GAS_Builder::Build bis.size " << gas.bis.size() << std::endl ; 
    Build(gas); 
}


BI GAS_Builder::MakeCustomPrimitivesBI_1NN(const Shape* sh, unsigned i )
{
    std::cout << "GAS_Builder::MakeCustomPrimitivesBI_1NN " << std::endl ; 

    unsigned primitiveIndexOffset = i ;  // without this optixGetPrimitiveIndex() would always give zero in 1NN mode
    const float* aabb = sh->kludge_outer_aabb > 0  ? sh->aabb : sh->aabb + i*6u ; 

    unsigned num_sbt_records = 1 ; 

    BI bi = {} ; 
    bi.mode = 0 ; 
    bi.flags = new unsigned[num_sbt_records];
    bi.sbt_index = new unsigned[num_sbt_records];
    bi.flags[0] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ; // p18: Each build input also specifies an array of OptixGeometryFlags, one for each SBT record.
    bi.sbt_index[0] = 0 ; 

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &bi.d_aabb ), 6*sizeof(float) ) );
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( bi.d_aabb ), aabb, 6*sizeof(float), cudaMemcpyHostToDevice ));

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &bi.d_sbt_index ), sizeof(unsigned)*num_sbt_records ) ); 
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( bi.d_sbt_index ), bi.sbt_index, sizeof(unsigned)*num_sbt_records, cudaMemcpyHostToDevice ) ); 

    bi.buildInput = {};
    bi.buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;  
    buildInputCPA.aabbBuffers = &bi.d_aabb ;  
    buildInputCPA.numPrimitives = 1 ;   
    buildInputCPA.strideInBytes = sizeof(float)*6  ; // stride between AABBs, 0-> sizeof(optixAabb)  
    buildInputCPA.flags = bi.flags;
    buildInputCPA.numSbtRecords = num_sbt_records ;  
    buildInputCPA.sbtIndexOffsetBuffer  = bi.d_sbt_index ;
    buildInputCPA.sbtIndexOffsetSizeInBytes  = sizeof(unsigned);
    buildInputCPA.sbtIndexOffsetStrideInBytes = sizeof(unsigned);
    buildInputCPA.primitiveIndexOffset = primitiveIndexOffset ;  // Primitive index bias, applied in optixGetPrimitiveIndex()
    return bi ; 
} 


/**
GAS_Builder::Build_11N GAS:BI:AABB  1:1:N  one BI with multiple AABB
------------------------------------------------------------------------
**/

void GAS_Builder::Build_11N( GAS& gas, const Shape* sh )
{
    std::cout << "GAS_Builder::Build_11N sh.num " << sh->num << std::endl ;  
    gas.sh = sh ; 

    BI bi = MakeCustomPrimitivesBI_11N( sh );
    gas.bis.push_back(bi); 

    Build(gas); 
}

BI GAS_Builder::MakeCustomPrimitivesBI_11N(const Shape* sh)
{
    std::cout << "GAS_Builder::MakeCustomPrimitivesBI_11N " << std::endl ; 
    
    BI bi = {} ; 
    bi.mode = 1 ; 
    unsigned num = sh->num ; 

    bi.flags = new unsigned[num];
    bi.sbt_index = new unsigned[num];
    for(unsigned i=0 ; i < num ; i++) bi.flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT ; 
    for(unsigned i=0 ; i < num ; i++) bi.sbt_index[i] = i ; 

    unsigned primitiveIndexOffset = 0 ; // offsets the normal 0,1,2,... result of optixGetPrimitiveIndex()  

    const float* aabb = sh->aabb ; 

    std::cout << "GAS_Builder::MakeCustomPrimitivesBI_11N dump aabb for layers: " << num << std::endl ; 
    for(unsigned i=0 ; i < num ; i++)
    { 
        std::cout << std::setw(4) << i << " : " ; 
        for(unsigned j=0 ; j < 6 ; j++)  
           std::cout << std::setw(10) << std::fixed << std::setprecision(3) << *(aabb + i*6 + j ) << " "  ; 
        std::cout << std::endl ; 
    }

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&bi.d_aabb), 6*sizeof(float)*num ));
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>(bi.d_aabb), aabb, 6*sizeof(float)*num, cudaMemcpyHostToDevice ));

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &bi.d_sbt_index ), sizeof(unsigned)*num ) ); 
    CUDA_CHECK( cudaMemcpy( reinterpret_cast<void*>( bi.d_sbt_index ), bi.sbt_index, sizeof(unsigned)*num, cudaMemcpyHostToDevice ) ); 

    bi.buildInput = {};
    bi.buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;  
    buildInputCPA.aabbBuffers = &bi.d_aabb ;  
    buildInputCPA.numPrimitives = num  ;   
    buildInputCPA.strideInBytes = sizeof(float)*6  ; // stride between AABBs, 0-> sizeof(optixAabb)  
    buildInputCPA.flags = bi.flags;                  // flags per sbt record 
    buildInputCPA.numSbtRecords = num ;              // number of sbt records available to sbt index offset override. 
    buildInputCPA.sbtIndexOffsetBuffer  = bi.d_sbt_index ;   // Device pointer to per-primitive local sbt index offset buffer, Every entry must be in range [0,numSbtRecords-1]
    buildInputCPA.sbtIndexOffsetSizeInBytes  = sizeof(unsigned);  // Size of type of the sbt index offset. Needs to be 0,     1, 2 or 4    
    buildInputCPA.sbtIndexOffsetStrideInBytes = sizeof(unsigned); // Stride between the index offsets. If set to zero, the offsets are assumed to be tightly packed.
    buildInputCPA.primitiveIndexOffset = primitiveIndexOffset ;  // Primitive index bias, applied in optixGetPrimitiveIndex()

    return bi ; 
} 


void GAS_Builder::Build(GAS& gas)   // static 
{ 
    std::cout << "GAS_Builder::Build" << std::endl ;  

    assert( gas.bis.size() > 0 ); 

    std::vector<OptixBuildInput> buildInputs ; 
    for(unsigned i=0 ; i < gas.bis.size() ; i++)
    {
        const BI& bi = gas.bis[i]; 
        buildInputs.push_back(bi.buildInput); 
    }

    std::cout 
        << "GAS_Builder::Build" 
        << " gas.bis.size " << gas.bis.size()
        << " buildInputs.size " << buildInputs.size()
        << std::endl 
        ;  

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = 
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION ;
    accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;

    OPTIX_CHECK( optixAccelComputeMemoryUsage( Ctx::context, 
                                               &accel_options, 
                                               buildInputs.data(), 
                                               buildInputs.size(), 
                                               &gas_buffer_sizes 
                                             ) );
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK( cudaMalloc( 
                reinterpret_cast<void**>( &d_temp_buffer_gas ), 
                gas_buffer_sizes.tempSizeInBytes 
                ) );

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );

    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_buffer_temp_output_gas_and_compacted_size ),
                compactedSizeOffset + 8
                ) );


    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    OPTIX_CHECK( optixAccelBuild( Ctx::context,
                                  0,                  // CUDA stream
                                  &accel_options,
                                  buildInputs.data(),
                                  buildInputs.size(),                  // num build inputs
                                  d_temp_buffer_gas,
                                  gas_buffer_sizes.tempSizeInBytes,
                                  d_buffer_temp_output_gas_and_compacted_size,
                                  gas_buffer_sizes.outputSizeInBytes,
                                  &gas.handle,
                                  &emitProperty,      // emitted property list
                                  1                   // num emitted properties
                                  ) );

    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );
    //CUDA_CHECK( cudaFree( (void*)d_aabb_buffer ) );

    size_t compacted_gas_size;
    CUDA_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &gas.d_buffer ), compacted_gas_size ) );

        // use handle as input and output
        OPTIX_CHECK( optixAccelCompact( Ctx::context, 
                                        0, 
                                        gas.handle, 
                                        gas.d_buffer, 
                                        compacted_gas_size, 
                                        &gas.handle ) );

        CUDA_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        gas.d_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

