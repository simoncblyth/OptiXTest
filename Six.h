#pragma once

#include <vector>
#include <optixu/optixpp_namespace.h>

struct Geo ; 
struct Foundry ; 
struct Grid ; 
struct Params ; 
struct Solid ; 

struct Six
{
    optix::Context context ;
    optix::Material material ;
    optix::Buffer pixels_buffer ; 
    optix::Buffer posi_buffer ; 
    std::vector<optix::Geometry> solids ; 
    std::vector<optix::Group>    assemblies ; 

    const Params* params ; 
    const char*   ptx_path ; 
    unsigned    entry_point_index ; 

    Six(const char* ptx_path, const Params* params_);  

    void initContext();
    void initPipeline();
    void setGeo(const Geo* geo);

    optix::GeometryInstance createGeometryInstance(unsigned solid_idx, unsigned identity);
    optix::Geometry         createSolidGeometry(const Foundry* foundry, unsigned solid_idx);
    optix::GeometryGroup    createSimple(const Geo* geo);
    
    void createSolids(const Foundry* foundry);
    void createGrids(const Geo* geo);
    optix::Group convertGrid(const Grid* gr);

    void launch();

    void save(const char* outdir) ; 


};
