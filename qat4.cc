// name=qat4 ; glm- ; gcc $name.cc -I. -I/usr/local/cuda/include -I$(glm-prefix) -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include "sutil_vec_math.h"

#include <string>
#include <vector>
#include <iostream>

#include "qat4.h"

#include <glm/mat4x4.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

struct points
{
    std::vector<std::string> vn ; 
    std::vector<glm::vec4>   vv ; 
    std::vector<float4>      ff ; 

    void add(float x, float y, float z, float w, const char* label )
    {
        ff.push_back(make_float4(x,y,z,w));  
        vv.push_back(glm::vec4(x,y,z,w)) ;
        vn.push_back(label); 
    } 

    void dump(const char* msg)
    {
        for(unsigned i=0 ; i < vn.size() ; i++)
        {
            std::cout 
                << vn[i] 
                << " vv: " << glm::to_string(vv[i]) 
                << " ff: " << ff[i] 
                << std::endl 
                ; 
        }
    }

    void test_mat_vec( const float* d )
    {
        dump16(d, "d:"); 
        glm::mat4 g = glm::make_mat4(d); 
        qat4 q(d); 

        std::cout << "g " << glm::to_string(g) << std::endl ;  
        std::cout << "q " << q << std::endl ;  

        for(unsigned i=0 ; i < vn.size() ; i++)
        {
            const glm::vec4& v = vv[i] ; 
            const float4& f = ff[i] ; 

            float4    qf = q * f ; 
            glm::vec4 gv = g * v ; 

            std::cout 
                << vn[i] 
                << " v: " << glm::to_string(v) 
                << " gv: " << glm::to_string(gv)
                << " qf: " << qf
                << std::endl 
                ; 

            assert( qf.x == gv.x ); 
            assert( qf.y == gv.y ); 
            assert( qf.z == gv.z ); 
            assert( qf.w == gv.w ); 
        }
    }


    void dump16( const float* f, const char* label)
    {
        std::cout << label << " " ; 
        for(unsigned i=0 ; i < 16 ; i++ ) std::cout << *(f+i) << " " ; 
        std::cout << std::endl; 
    }
};

glm::mat4 make_transform(float tx, float ty, float tz, float sx, float sy, float sz, float ax, float ay, float az, float degrees )
{
    float radians = (degrees/180.f)*glm::pi<float>() ; 

    glm::mat4 m(1.f); 
    m = glm::translate(  m, glm::vec3(tx, ty, tz) );             std::cout << " t " << glm::to_string(m) << std::endl ;  
    m = glm::rotate(     m, radians, glm::vec3(ax, ay, az)  );   std::cout << " rt " << glm::to_string(m) << std::endl ;  
    m = glm::scale(      m, glm::vec3(sx, sy, sz) );             std::cout << " srt " << glm::to_string(m) << std::endl ;

    return m ;
}

int main(int argc, char** argv)
{
    points p ; 
    p.add(0.f, 0.f, 0.f, 1.f, "po"); 
    p.add(1.f, 0.f, 0.f, 1.f, "px"); 
    p.add(0.f, 1.f, 0.f, 1.f, "py"); 
    p.add(0.f, 0.f, 1.f, 1.f, "pz"); 
    p.dump("points.p"); 


    float tx = 100.f ;  
    float ty = 200.f ;  
    float tz = 300.f ;  

    float sx = 10.f ;  
    float sy = 10.f ;  
    float sz = 10.f ;  

    float ax = 0.f ;  
    float ay = 0.f ;  
    float az = 1.f ;  

    float degrees = 45.0 ; 

    glm::mat4 srt = make_transform(tx, ty, tz, sx, sy, sz, ax, ay, az, degrees); 
    p.test_mat_vec( glm::value_ptr(srt) ); 

    return 0 ; 
}
