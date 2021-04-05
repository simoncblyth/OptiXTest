// name=sizeof ; gcc $name.cc -std=c++11 -I/usr/local/cuda/include -lstdc++ -o /tmp/$name && /tmp/$name
#include <iostream>
#include <iomanip>
#include <vector>

#include "sutil_vec_math.h"
#include "Node.h"


int main(int argc, char** argv)
{
    std::cout 
        << std::setw(30) << "sizeof(float)"   
        << " (dec)" << std::dec << sizeof(float) 
        << " (hex)" << std::hex << sizeof(float) 
        << std::endl 
        << std::setw(30) << "sizeof(float)*6" 
        << " (dec) " <<  std::dec << sizeof(float)*6 
        << " (hex) " <<  std::hex << sizeof(float)*6 
        << std::dec
        << std::endl 
        ;  


    std::vector<Node> nds ; 
    for(unsigned i=0 ; i < 10 ; i++)
    {
        float v = 100.f + i*10.f ; 
        Node nd = {} ; 
        nd.setAABB( -v, -v, -v, v, v, v ); 
        nds.push_back(nd) ;
    }  

    unsigned size_in_floats = 6 ; 
    unsigned offset_in_floats = 8 ; 
    unsigned stride_in_floats = 16 ; 

    std::vector<float> tmp(size_in_floats*nds.size()) ; 
    for(unsigned i=0 ; i < nds.size() ; i++) 
    {
        float* dst = tmp.data() + 6*i ;  
        const float* src = (float*)nds.data() + offset_in_floats + stride_in_floats*i ;  
        memcpy(dst, src,  sizeof(float)*size_in_floats );  
    }

    for(unsigned j=0 ; j < tmp.size() ; j++)
    {
        if( j % size_in_floats == 0 ) std::cout << std::endl ; 
        std::cout << tmp[j] << " " ; 
    }
    std::cout << std::endl ; 


    return 0 ; 
} 
