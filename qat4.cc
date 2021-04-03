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


#include <glm/gtc/random.hpp>

const float EPSILON = 1e-4 ; 


bool check( const float& a, const float& b,  const char* msg  )
{
    float cf = std::abs(a - b) ; 
    bool chk = cf < EPSILON ; 
    if(!chk) 
        std::cout 
            << msg
            << " cf "
            << std::fixed << std::setprecision(8) << cf
            << " ( "
            << std::fixed << std::setprecision(8) << a 
            << "," 
            << std::fixed << std::setprecision(8) << b 
            << ") "
            << std::endl 
            ; 
     return chk ; 
}

void check( const glm::vec4& a , const glm::vec4& b, const char* msg )
{
    bool chk = check(a.x,b.x,"x") && check(a.y,b.y,"y") && check(a.z,b.z,"z") && check(a.w,b.w,"w") ;
    if(!chk) std::cout << msg << std::endl ;  
    assert(chk); 
}
void check( const float4& a , const float4& b, const char* msg )
{
    bool chk = check(a.x,b.x,"x") && check(a.y,b.y,"y") && check(a.z,b.z,"z") && check(a.w,b.w,"w") ;
    if(!chk) std::cout << msg << std::endl ;  
    assert(chk); 
}
void check( const float4& a , const glm::vec4& b, const char* msg )
{
    bool chk = check(a.x,b.x,"x") && check(a.y,b.y,"y") && check(a.z,b.z,"z") && check(a.w,b.w,"w") ;
    if(!chk) std::cout << msg << std::endl ;  
    assert(chk); 
}
void check( const glm::vec4& a , const float4& b, const char* msg  )
{
    bool chk = check(a.x,b.x,"x") && check(a.y,b.y,"y") && check(a.z,b.z,"z") && check(a.w,b.w,"w") ;
    if(!chk) std::cout << msg << std::endl ;  
    assert(chk); 
}




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


    void test_multiply( const float* d )
    {
        dump16(d, "d:"); 

        glm::mat4 g = glm::make_mat4(d); 
        glm::mat4 t = glm::transpose(g); 
        std::cout << "g " << glm::to_string(g) << std::endl ;  
        std::cout << "t " << glm::to_string(t) << std::endl ;  

        qat4 q(glm::value_ptr(g)); 
        qat4 r(glm::value_ptr(t));       // r is tranposed(q)
        std::cout << "q " << q << std::endl ;  
        std::cout << "r " << r << std::endl ;  


        for(unsigned i=0 ; i < vn.size() ; i++)
        {
            const glm::vec4& v = vv[i] ; 
            const float4&    f = ff[i] ; 

            //            mat*vec
            float4    qf = q * f ;   
            glm::vec4 gv = g * v ; 
            check(qf, gv, "qf == gv : glm/qat consistency for mat*vec "); 

            //            vec*transposed(mat)
            float4    fr = f * r ; 
            glm::vec4 vt = v * t ; 
            check(fr, vt, "fr == vt : glm/qat consisency of vec*transposed(mat)"); 
            check(qf, fr, "qf == fr : qat consistency  mat*vec == vec*transposed(mat) ");    
            check(gv, vt, "gv == vt : glm consistency  mat*vec == vec*transposed(mat)");    

            //            vec*mat
            float4    fq = f * q ; 
            glm::vec4 vg = v * g ; 
            check(fq, vg, "fq == vg : glm/qat consisency of vec*mat"); 

            //transposed(mat)*vec  
            float4    rf = r * f; 
            glm::vec4 tv = t * v ; 
            check(rf, tv,  "rf == tv : glm/qat consistency  vec*mat == transposed(mat)*vec ");    
            check(fq, rf,  "fq == rf : qat consistency      vec*mat == transposed(mat)*vec ");
            check(vg, tv,  "vg == tv : glm consistency      vec*mat == transposted(mat)*vec " );  


            std::cout 
                << vn[i] 
                << " v: " << glm::to_string(v) 
                << " gv: " << glm::to_string(gv)
                << " qf: " << qf
                << std::endl 
                ; 

            std::cout 
                << vn[i] 
                << " v: " << glm::to_string(v) 
                << " vg: " << glm::to_string(vg)
                << " fq: " << fq
                << std::endl 
                ; 

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

glm::mat4 make_transform_0()
{
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
    return srt ; 
}

glm::mat4 make_transform_1()
{
    glm::vec3 t = glm::sphericalRand(20000.f); 
    glm::vec3 a = glm::sphericalRand(1.f); 
    //glm::vec3 s(  glm::linearRand(1.f, 100.f), glm::linearRand(1.f, 100.f), glm::linearRand(1.f, 100.f) );  
    glm::vec3 s(1.f,1.f,1.f );  
    float degrees = glm::linearRand(0.f, 360.f ); 
    glm::mat4 srt = make_transform(t.x, t.y, t.z, s.x, s.y, s.z, a.x, a.y, a.z, degrees); 
    return srt ; 
}


void test_multiply()
{
    points p ; 
    p.add(0.f, 0.f, 0.f, 1.f, "po"); 
    p.add(1.f, 0.f, 0.f, 1.f, "px"); 
    p.add(0.f, 1.f, 0.f, 1.f, "py"); 
    p.add(0.f, 0.f, 1.f, 1.f, "pz"); 

    for(unsigned i=0 ; i < 100 ; i++)
    {
        glm::vec3 r = glm::sphericalRand(20000.f); 
        p.add( r.x, r.y, r.z, 1.0, "r" ); 
    } 
    p.dump("points.p"); 
    glm::mat4 t0 = make_transform_0(); 
    p.test_multiply( glm::value_ptr(t0) ); 

/*
    for(unsigned i=0 ; i < 100 ; i++)
    {
        glm::mat4 t1 = make_transform_1(); 
        p.test_multiply( glm::value_ptr(t1) ); 
    }
*/
}


int main(int argc, char** argv)
{
    glm::mat4 m(1.f); 
    glm::vec3 s(1.f, 1.f, 1.f) ; 

    m = glm::scale(m, s ); 
    std::cout << glm::to_string( m) << std::endl ; 

    qat4 q(glm::value_ptr(m)); 
    float4 isect = make_float4(10.f, 10.f, 10.f, 42.f ); 
    q.right_multiply_inplace( isect, 0.f ) ;
    printf("isect: (%10.4f, %10.4f, %10.4f, %10.4f) \n", isect.x, isect.y, isect.z, isect.w ); 

    return 0 ; 
}
