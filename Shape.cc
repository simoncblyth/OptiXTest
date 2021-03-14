#include <sstream>
#include <vector_types.h>

#include "OpticksCSG.h"
#include "Sys.h"
#include "Shape.h"
#include "NP.hh"

#include <glm/gtc/type_ptr.hpp>

unsigned Shape::Type(char typ)  // static 
{
    unsigned t(CSG_ZERO) ;  
    switch(typ)
    {
       case 'S': t = CSG_SPHERE ; break ; 
       case 'B': t = CSG_BOX    ; break ; 
    }
    return t ; 
}

bool Shape::is1NN() const { return gas_bi_aabb == GBA_1NN ; }
bool Shape::is11N() const { return gas_bi_aabb == GBA_11N ; }


Shape::Shape(const char typ, float sz)
    :
    num(1),
    typs(new char[num]),
    kludge_outer_aabb(0),
    gas_bi_aabb(GBA_1NN)
{
    typs[0] = typ ; 
    std::vector<float> szs = { sz } ; 
    init(szs); 
}

Shape::Shape(const char* typs_, const std::vector<float>& szs)
    :
    num(szs.size()),
    typs(new char[num]),
    kludge_outer_aabb(0),
    gas_bi_aabb(GBA_1NN)
{
    size_t len = strlen(typs_); 
    for(unsigned i=0 ; i < num ; i++) typs[i] = i < len ? typs_[i] : typs_[0] ;    // duplicate the first typ, if run out 
    init(szs); 
}

Shape::~Shape()
{
    delete [] typs ; 
}


void Shape::init( const std::vector<float>& szs )
{
    for(unsigned i=0 ; i < szs.size() ; i++) assert(szs[i] <= szs[0] ) ; 
    int ni = num ; 
    for(int i=0 ; i < ni ; i++)
    {
        float size = szs[i] ;  
        char type = typs[i] ; 
        switch(type)
        {
           case 'S': add_sphere(size)  ; break ; 
           case 'Z': add_zsphere(size) ; break ; 
        }
    }
}

void Shape::add_sphere(float radius)
{
    Node nd ; 
    nd.q0.f = {0.f, 0.f, 0.f, radius} ; 
    nd.q1.i = {0,0,0,0} ; 
    nd.q2.u = {0,0,0,CSG_SPHERE} ; 
    nd.q3.i = {0,0,0,0} ; 

    node.push_back(nd); 

    int num_node = 1 ; 
    add_prim(num_node);  

    AABB bb ; 
    bb.mn = {-radius, -radius, -radius }; 
    bb.mx = { radius,  radius,  radius }; 
    aabb.push_back(bb); 
}

void Shape::add_zsphere(float radius)
{
    Node nd ; 

    float3 center = {0.f, 0.f, 0.f } ;  
    float2 zdelta = {-radius/2.f , +radius/2.f } ; 

    const float zmax = center.z + zdelta.y ;
    const float zmin = center.z + zdelta.x ;

    nd.q0.f = { center.x, center.y, center.z, radius} ; 
    nd.q1.f = { zdelta.x, zdelta.y, 0,0} ; 
    nd.q2.u = {0,0,0, CSG_ZSPHERE} ; 
    nd.q3.i = {0,0,0,0} ; 

    node.push_back(nd); 

    int num_node = 1 ; 
    add_prim(num_node);  

    AABB bb ; 
    bb.mx = { center.x+radius, center.y+radius, zmax }; 
    bb.mn = { center.x-radius, center.y-radius, zmin }; 
    aabb.push_back(bb); 
}



void Shape::add_prim(int num_node)
{
    int node_offset = node.size() ;
    int tran_offset = tran.size() ;
    int plan_offset = plan.size() ;

    glm::ivec4 pr(node_offset, num_node, tran_offset, plan_offset)  ; 
    prim.push_back(pr); 
}

const glm::ivec4& Shape::get_prim_(unsigned prim_idx) const 
{
    assert( prim_idx < prim.size() ); 
    return prim[prim_idx] ; 
}
unsigned Shape::get_num_node(unsigned prim_idx) const 
{
    const glm::ivec4& pr = get_prim_(prim_idx);
    return pr.y ; 
}
int* Shape::get_prim(unsigned prim_idx) const
{
    assert( prim_idx < num ); 
    return (int*)prim.data() + prim_idx*prim_size ; 
}
const Node* Shape::get_node(unsigned idx) const  
{
    assert( idx < num ); 
    const Node* n = node.data() + idx ; 
    return n ; 
}
const AABB* Shape::get_aabb(unsigned idx) const
{
    assert( idx < num ); 
    const AABB* bb = aabb.data() + idx ; 
    return bb ; 
}
char Shape::get_type(unsigned idx) const
{
    assert( idx < num ); 
    return typs[idx] ; 
}

std::string Shape::desc(unsigned idx) const 
{  
    std::stringstream ss ; 
    ss << " idx: " << idx  ;
    ss << " typ: " << get_type(idx)  ;
    ss << " kludge_outer_aabb: " << kludge_outer_aabb ;  
    ss << " gas_bi_aabb " << gas_bi_aabb ; 
    ss << " aabb: " 
       << aabb[idx].mn.x << " "  
       << aabb[idx].mn.y << " "  
       << aabb[idx].mn.z << " "  
       << aabb[idx].mx.x << " "  
       << aabb[idx].mx.y << " "  
       << aabb[idx].mx.z << " "  
       ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string Shape::desc() const 
{
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num ; i++ ) ss << desc(i) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

void Shape::write(const char* base, const char* rel, unsigned idx) const 
{
    std::stringstream ss ;   
    ss << base << "/" << rel << "/" << idx << "/" ; 
    std::string dir = ss.str();   
    NP::Write(dir.c_str(), "aabb.npy", (float*)aabb.data(), num, 2, 3 ); 
    NP::Write(dir.c_str(), "node.npy", (float*)node.data(), num, 4, 4 ); 
    NP::Write(dir.c_str(), "prim.npy",   (int*)prim.data(), num, 4 ); 
}



void Shape::Dump(const float* f, const int ni, const char* label)
{
    if(label) std::cout << label << std::endl ; 
    int nj = 4 ;  
    int nk = 4 ; 
    for(int i=0 ; i < ni ; ++i )
    {
        std::cout << "(" << i << ")" << std::endl ; 
        for(int j=0 ; j < nj ; j++) 
        {
            for(int k=0 ; k < nk ; k++) std::cout << f[i*nj*nk+j*nk+k] << " " ;  
            std::cout << std::endl ; 
        }
        std::cout << std::endl ; 
    }
}
void Shape::Dump(const glm::mat4& nd ) // static 
{
    const float* f = glm::value_ptr(nd);  
    Dump(f, 1, "mat4") ;
}
void Shape::Dump(const std::vector<glm::mat4>& nds)
{
    float* f = (float*)nds.data() ;
    Dump(f, nds.size(), "vec.mat4" ) ;
}






