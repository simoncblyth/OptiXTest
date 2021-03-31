/**
OptiXTest
==========

**/
#include <iostream>
#include <cstdlib>

#include <optix.h>

#if OPTIX_VERSION < 70000
#else
#include <optix_stubs.h>
#endif

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "Util.h"
#include "Geo.h"
#include "Frame.h"
#include "Params.h"

#if OPTIX_VERSION < 70000
#include "Six.h"
#else
#include "Ctx.h"
#include "CUDA_CHECK.h"   
#include "OPTIX_CHECK.h"   
#include "PIP.h"
#include "SBT.h"

#include "Foundry.h"
#endif


struct AS ; 

int main(int argc, char** argv)
{
#if OPTIX_VERSION < 70000
    const char* ptxname = "OptiX6Test" ; 
#else
    const char* ptxname = "OptiX7Test" ; 
#endif
    std::cout << ptxname << std::endl ; 

    const char* prefix = getenv("PREFIX");  assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    const char* outdir = getenv("OUTDIR");  assert( outdir && "expecting OUTDIR envvar " );

    const char* cmake_target = "OptiXTest" ; 
    const char* ptx_path = Util::PTXPath( prefix, cmake_target, ptxname ) ; 
    std::cout << " ptx_path " << ptx_path << std::endl ; 

    bool small = false ;  
    unsigned width = small ? 512u : 1024u ; 
    unsigned height = small ? 384u : 768u ; 
    unsigned depth = 1u ; 
    unsigned cameratype = Util::GetEValue<unsigned>("CAMERATYPE", 0u ); 


    Foundry foundry ; 
    foundry.makeSphere(); 

    Geo geo ;  
    geo.write(outdir);  

    glm::vec3 eye_model ; 
    Util::GetEVec(eye_model, "EYE", "-1.0,-1.0,1.0"); 

    float top_extent = geo.getTopExtent() ;  
    glm::vec4 ce(0.f,0.f,0.f, top_extent*1.4f );   // defines the center-extent of the region to view
    glm::vec3 eye,U,V,W  ;
    Util::GetEyeUVW( eye_model, ce, width, height, eye, U, V, W ); 

    Params params ; 
    params.setView(eye, U, V, W, geo.tmin, geo.tmax, cameratype ); 
    params.setSize(width, height, depth); 

    foundry.upload();   // uploads nodes, planes, transforms
    params.node = foundry.d_node ; 
    params.plan = foundry.d_plan ; 
    params.tran = foundry.d_tran ; 
    

#if OPTIX_VERSION < 70000

    Six six(ptx_path, &params); 
    six.setGeo(&geo); 
    six.launch(); 
    six.save(outdir); 

#else
    Ctx ctx(&params) ;

    PIP pip(ptx_path); 
    SBT sbt(&pip);
    sbt.setGeo(&geo);    // creates GAS, IAS, SBT records 


    AS* top = sbt.getTop(); 
    params.handle = top->handle ; 

    Frame frame(params.width, params.height, params.depth); 
    params.pixels = frame.getDevicePixels(); 
    params.isect  = frame.getDeviceIsect(); 
    ctx.uploadParams();  

    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip.pipeline, stream, ctx.d_param, sizeof( Params ), &(sbt.sbt), frame.width, frame.height, frame.depth ) );
    CUDA_SYNC_CHECK();

    frame.download(); 
    frame.write(outdir);  
#endif

    return 0 ; 
}
