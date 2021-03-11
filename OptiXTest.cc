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
#include <optixu/optixpp_namespace.h>
#include "SPPM.h" 
#include "NP.hh"
#else
#include "Ctx.h"
#include "CUDA_CHECK.h"   
#include "OPTIX_CHECK.h"   
#include "PIP.h"
#include "SBT.h"
#endif

/**
In [13]: np.unique( a[:,:,3].view(np.int32), return_counts=True )
Out[13]: (array([ 0, 42], dtype=int32), array([305534, 480898]))
**/



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


#if OPTIX_VERSION < 70000
    optix::Context context = optix::Context::create();
    context->setRayTypeCount(1);
    context->setPrintEnabled(true);
    context->setPrintBufferSize(4096);
    context->setEntryPointCount(1);
    unsigned entry_point_index = 0u ;
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path , "raygen" ));
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( ptx_path , "miss" ));

    optix::Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount( 1u );
    sphere->setBoundingBoxProgram( context->createProgramFromPTXFile( ptx_path , "bounds" ) );
    sphere->setIntersectionProgram( context->createProgramFromPTXFile( ptx_path , "intersect" ) ) ; 

    optix::Material mat = context->createMaterial();
    mat->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path, "closest_hit" ));

    unsigned identity = 42u ; 
    optix::GeometryInstance pergi = context->createGeometryInstance() ;
    pergi->setMaterialCount(1);
    pergi->setMaterial(0, mat );
    pergi->setGeometry(sphere);
    pergi["identity"]->setUint(identity);

    optix::GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount(1);
    gg->setChild( 0, pergi );
    gg->setAcceleration( context->createAcceleration("Trbvh") );

    context["top_object"]->set( gg );

    float near = 0.1f ; 
    context[ "scene_epsilon"]->setFloat( near );  
    context[ "eye"]->setFloat( params.eye.x, params.eye.y, params.eye.z  );  
    context[ "U"  ]->setFloat( params.U.x, params.U.y, params.U.z  );  
    context[ "V"  ]->setFloat( params.V.x, params.V.y, params.V.z  );  
    context[ "W"  ]->setFloat( params.W.x, params.W.y, params.W.z  );  
    context[ "radiance_ray_type"   ]->setUint( 0u );  

    optix::Buffer pixels_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, params.width, params.height);
    context["pixels_buffer"]->set( pixels_buffer );
    optix::Buffer posi_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, params.width, params.height);
    context["posi_buffer"]->set( posi_buffer );

    context->launch( entry_point_index , params.width, params.height  );  

    int channels = 4 ; 
    SPPM_write(outdir, "pixels.ppm",  (unsigned char*)pixels_buffer->map(), channels,  width, height, true );
    pixels_buffer->unmap(); 

    NP::Write(outdir, "posi.npy",  (float*)posi_buffer->map(), height, width, 4 );
    posi_buffer->unmap(); 

#else
    Ctx ctx(&params) ;

    PIP pip(ptx_path); 
    SBT sbt(&pip);
    sbt.setGeo(&geo); 
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
