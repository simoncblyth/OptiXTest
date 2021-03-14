#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Params.h"
#include "InstanceId.h"
#include "Geo.h"
#include "Shape.h"
#include "Grid.h"
#include "Six.h"
#include "SPPM.h" 
#include "NP.hh"
    
Six::Six(const char* ptx_path_, const Params* params_)
    :
    context(optix::Context::create()),
    material(context->createMaterial()),
    params(params_),
    ptx_path(strdup(ptx_path_)),
    entry_point_index(0u) 
{
    initContext();
    initPipeline(); 
}

void Six::initContext()
{
    context->setRayTypeCount(1);
    context->setPrintEnabled(true);
    context->setPrintBufferSize(4096);
    context->setPrintLaunchIndex(0); 
    context->setEntryPointCount(1);

    context[ "tmin"]->setFloat( params->tmin );  
    context[ "eye"]->setFloat( params->eye.x, params->eye.y, params->eye.z  );  
    context[ "U"  ]->setFloat( params->U.x, params->U.y, params->U.z  );  
    context[ "V"  ]->setFloat( params->V.x, params->V.y, params->V.z  );  
    context[ "W"  ]->setFloat( params->W.x, params->W.y, params->W.z  );  
    context[ "radiance_ray_type"   ]->setUint( 0u );  

    pixels_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, params->width, params->height);
    context["pixels_buffer"]->set( pixels_buffer );
    posi_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, params->width, params->height);
    context["posi_buffer"]->set( posi_buffer );
}

void Six::initPipeline()
{
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path , "raygen" ));
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( ptx_path , "miss" ));

    material->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path, "closest_hit" ));
}

void Six::setGeo(const Geo* geo)
{
    unsigned num_shape = geo->getNumShape(); 
    std::cout << "Six::setGeo num_shape " << num_shape << std::endl ;  
    createShapes(geo); 

    //optix::GeometryGroup gg = createSimple(geo); 
    createGrids(geo); 

    const char* spec = geo->top ;  
    char c = spec[0]; 
    assert( c == 'i' || c == 'g' );  
    int idx = atoi( spec + 1 );  

    std::cout << "Six::setGeo spec " << spec << std::endl ; 
    if( c == 'i' )
    {
        assert( idx < assemblies.size() ); 
        optix::Group grp = assemblies[idx]; 
        context["top_object"]->set( grp );
    }
    else if( c == 'g' )
    {
        assert( idx < shapes.size() ); 

        optix::GeometryGroup gg = context->createGeometryGroup();
        gg->setChildCount(1);
     
        unsigned identity = 1u + idx ;  
        optix::GeometryInstance pergi = createGeometryInstance(idx, identity); 
        gg->setChild( 0, pergi );
        gg->setAcceleration( context->createAcceleration("Trbvh") );

        context["top_object"]->set( gg );
    }
}

void Six::createShapes(const Geo* geo)
{
    unsigned num_shape = geo->getNumShape(); 
    std::cout << "Six::createShapes num_shape " << num_shape << std::endl ;  

    for(unsigned i=0 ; i < num_shape ; i++)
    {
        const Shape* sh = geo->getShape(i) ;    
        optix::Geometry shape = createGeometry(sh); 
        shapes.push_back(shape); 
    }
}

optix::GeometryGroup Six::createSimple(const Geo* geo)
{
    unsigned num_shape = geo->getNumShape(); 
    optix::GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount(num_shape);
    for(unsigned i=0 ; i < num_shape ; i++)
    {
        unsigned identity = 1u + i ;  
        optix::GeometryInstance pergi = createGeometryInstance(i, identity); 
        gg->setChild( i, pergi );
    }
    gg->setAcceleration( context->createAcceleration("Trbvh") );
    return gg ; 
}

void Six::createGrids(const Geo* geo)
{
    unsigned num_grid = geo->getNumGrid(); 
    for(unsigned i=0 ; i < num_grid ; i++)
    {
        const Grid* gr = geo->getGrid(i) ;    
        optix::Group assembly = convertGrid(gr); 
        assemblies.push_back(assembly); 
    }
}


/**
Six::convertGrid
------------------

Identity interpretation needs to match what IAS_Builder::Build is doing 

**/

optix::Group Six::convertGrid(const Grid* gr)
{
    unsigned num_tr = gr->trs.size() ; 
    std::cout << "Six::convertGrid num_tr " << num_tr << std::endl ; 
    assert( num_tr > 0); 

    const char* accel = "Trbvh" ; 
    optix::Acceleration instance_accel = context->createAcceleration(accel);
    optix::Acceleration assembly_accel  = context->createAcceleration(accel);

    optix::Group assembly = context->createGroup();
    assembly->setChildCount( num_tr );
    assembly->setAcceleration( assembly_accel );  

    const float* vals =   (float*)gr->trs.data() ;

    for(unsigned i=0 ; i < num_tr ; i++)
    {
        glm::mat4 mat(1.0f) ; 
        memcpy( glm::value_ptr(mat), (void*)(vals + i*16), 16*sizeof(float));
        
        glm::mat4 imat = glm::transpose(mat);

        glm::uvec4 idv ; // after transposiing the last row contains the identity info 
        memcpy( glm::value_ptr(idv), &imat[3], 4*sizeof(float) ); 

        imat[3].x = 0.f ; 
        imat[3].y = 0.f ; 
        imat[3].z = 0.f ; 
        imat[3].w = 1.f ; 


        unsigned identity = idv.x ; 
        unsigned ins_idx ; 
        unsigned gas_idx ; 
        InstanceId::Decode( ins_idx, gas_idx, identity );

        optix::Transform xform = context->createTransform();

        bool transpose = false ; 
        xform->setMatrix(transpose, glm::value_ptr(imat), 0); 
        assembly->setChild(i, xform);

        optix::GeometryInstance pergi = createGeometryInstance(gas_idx, identity ); 
        optix::GeometryGroup perxform = context->createGeometryGroup();
        perxform->addChild(pergi); 
        perxform->setAcceleration(instance_accel) ; 

        xform->setChild(perxform);
    }
    return assembly ;
}

optix::GeometryInstance Six::createGeometryInstance(unsigned shape_idx, unsigned identity)
{
    std::cout 
        << "Six::createGeometryInstance"
        << " shape_idx " << shape_idx
        << " identity " << identity
        << " identity.hex " << std::hex <<  identity << std::dec
        << std::endl 
        ;   

    optix::Geometry shape = shapes[shape_idx]; 

    optix::GeometryInstance pergi = context->createGeometryInstance() ;
    pergi->setMaterialCount(1);
    pergi->setMaterial(0, material );
    pergi->setGeometry(shape);
    pergi["identity"]->setUint(identity);

    return pergi ; 
}

optix::Geometry Six::createGeometry(const Shape* sh)
{
    optix::Geometry shape = context->createGeometry();
    shape->setBoundingBoxProgram( context->createProgramFromPTXFile( ptx_path , "bounds" ) );
    shape->setIntersectionProgram( context->createProgramFromPTXFile( ptx_path , "intersect" ) ) ; 

    std::vector<float> shape_array ; 

    std::cout << "Six::createGeometry sh.num " << sh->num << " sizes: " ; 
    for(unsigned i=0 ; i < sh->num ; i++)
    {
        float size = sh->get_size(i); 
        shape_array.push_back( 0.f ); 
        shape_array.push_back( 0.f ); 
        shape_array.push_back( 0.f ); 
        shape_array.push_back( size );

        std::cout << size << " " ; 
    }
    std::cout << std::endl ;  

    unsigned num_prim = sh->num ; 
    shape->setPrimitiveCount( num_prim );

    optix::Buffer shape_buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, num_prim );
    memcpy( shape_buffer->map(), shape_array.data(), sizeof(float)*shape_array.size() ); 
    shape_buffer->unmap() ; 
    shape["shape_buffer"]->set( shape_buffer );
  
    return shape ; 
}

void Six::launch()
{
    context->launch( entry_point_index , params->width, params->height  );  
}

void Six::save(const char* outdir) 
{
    int channels = 4 ; 
    SPPM_write(outdir, "pixels.ppm",  (unsigned char*)pixels_buffer->map(), channels,  params->width, params->height, true );
    pixels_buffer->unmap(); 

    NP::Write(outdir, "posi.npy",  (float*)posi_buffer->map(), params->height, params->width, 4 );
    posi_buffer->unmap(); 
}

