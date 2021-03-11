#include <iostream>
#include <iomanip>

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include "sutil_Exception.h"   // CUDA_CHECK OPTIX_CHECK


#include "NP.hh"
#include "Util.h"
#include "Frame.h"
#include "SPPM.h"

#define SIMG_IMPLEMENTATION 1 
#include "SIMG.hh"


Frame::Frame(unsigned width_, unsigned height_, unsigned depth_)
    :
    width(width_),
    height(height_),
    depth(depth_)
{
    init();
}

void Frame::init()
{
    init_pixels(); 
    init_isect(); 
}


void Frame::init_pixels()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_pixels ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_pixels ),
                width*height*sizeof(uchar4)
                ) );
}

uchar4* Frame::getDevicePixels() const 
{
    return d_pixels ; 
}


void Frame::init_isect()
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_isect ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &d_isect ),
                width*height*sizeof(float4)
                ) );
}
float4* Frame::getDeviceIsect() const 
{
    return d_isect ; 
}





void Frame::download()
{
    download_pixels();  
    download_isect();  
}

void Frame::download_pixels()
{
    pixels.resize(width*height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( pixels.data() ),
                d_pixels,
                width*height*sizeof(uchar4),
                cudaMemcpyDeviceToHost
    ));
}

void Frame::download_isect()
{
    isect.resize(width*height);  
    CUDA_CHECK( cudaMemcpy(
                static_cast<void*>( isect.data() ),
                d_isect,
                width*height*sizeof(float4),
                cudaMemcpyDeviceToHost
    ));
}


void Frame::write(const char* outdir) const 
{
    std::cout << "Frame::write " << outdir << std::endl ; 
    bool yflip = false ; 
    writePPM(outdir, "pixels.ppm", yflip );  
    int quality = Util::GetEValue<int>("QUALITY", 50); 
    writeJPG(outdir, "pixels.jpg", quality);  
    writeNP(  outdir, "posi.npy" );
}

void Frame::writePPM(const char* dir, const char* name, bool yflip ) const 
{
    std::cout << "Frame::writePPM " << dir << "/" << name << std::endl ; 
    SPPM_write( dir, name, pixels.data(), width, height, yflip );
}
void Frame::writePNG(const char* dir, const char* name) const 
{
    int channels = 4 ; 
    const unsigned char* data = (const unsigned char*)pixels.data();  
    SIMG img(int(width), int(height), channels,  data ); 
    img.writePNG(dir, name); 
}
void Frame::writeJPG(const char* dir, const char* name, int quality) const 
{
    int channels = 4 ; 
    const unsigned char* data = (const unsigned char*)pixels.data();  
    SIMG img(int(width), int(height), channels,  data ); 
    img.writeJPG(dir, name, quality); 
}

void Frame::writeNP( const char* dir, const char* name) const 
{
    std::cout << "Frame::writeNP " << dir << "/" << name << std::endl ; 
    NP::Write(dir, name, getIntersectData(), height, width, 4 );
}

float* Frame::getIntersectData() const
{
    return (float*)isect.data();
}



