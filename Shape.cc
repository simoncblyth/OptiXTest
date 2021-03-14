#include <sstream>
#include <vector_types.h>

#include "Sys.h"
#include "Shape.h"
#include "NP.hh"

#include <glm/gtc/type_ptr.hpp>

unsigned Shape::Type(char typ)  // static 
{
    unsigned t(ZERO) ;  
    switch(typ)
    {
       case 'S': t = SPHERE ; break ; 
       case 'B': t = BOX    ; break ; 
    }
    return t ; 
}

bool Shape::is1NN() const { return gas_bi_aabb == GBA_1NN ; }
bool Shape::is11N() const { return gas_bi_aabb == GBA_11N ; }


Shape::Shape(const char typ, float sz)
    :
    num(1),
    typs(new char[num]),
    param(new float[4*num]),
    aabb(new float[6*num]),
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
    param(new float[4*num]),
    aabb(new float[6*num]),
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
    delete [] param ; 
    delete [] aabb ; 
}


void Shape::init( const std::vector<float>& szs )
{
    for(unsigned i=0 ; i < szs.size() ; i++) assert(szs[i] <= szs[0] ) ; 

    int ni = num ; 

    for(int i=0 ; i < ni ; i++)
    {
        float size = szs[i] ;  
        char type = typs[i] ; 

        aabb[0+6*i] = -size ; 
        aabb[1+6*i] = -size ; 
        aabb[2+6*i] = -size ; 
        aabb[3+6*i] =  size ; 
        aabb[4+6*i] =  size ; 
        aabb[5+6*i] =  size ; 

        param[0+4*i] = size ; 
        param[1+4*i] = 0.f ; 
        param[2+4*i] = 0.f ; 
        param[3+4*i] = Sys::unsigned_as_float(type) ; 
    }

    for(int i=0 ; i < ni ; i++)
    {
        float size = szs[i] ;  

        Node nd ; 
        nd.q0.f = {0.f, 0.f, 0.f, size} ; 
        nd.q1.i = {0,0,0,0} ; 
        nd.q2.i = {0,0,0,0} ; 
        nd.q3.i = {0,0,0,0} ; 

        int num_node = 1 ; 
        add_prim( num_node );  

        //Shape::Dump(nd); 
        node.push_back(nd); 
    }
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
const Node* Shape::get_node(unsigned idx) const  // hmm for variable tree sizes in each layer ?
{
    assert( idx < num ); 

    const Node* n = node.data() + idx ; 
    return n ; 
}






/**
struct Prim 
{
    __device__ int partOffset() const { return  q0.i.x ; } 
    __device__ int numParts()   const { return  q0.i.y < 0 ? -q0.i.y : q0.i.y ; } 
    __device__ int tranOffset() const { return  q0.i.z ; } 
    __device__ int planOffset() const { return  q0.i.w ; } 
    __device__ int primFlag()   const { return  q0.i.y < 0 ? CSG_FLAGPARTLIST : CSG_FLAGNODETREE ; } 

    quad q0 ; 

};

**/



float* Shape::get_aabb(unsigned idx) const
{
    assert( idx < num ); 
    return aabb + idx*6 ; 
}
float* Shape::get_param(unsigned idx) const
{
    assert( idx < num ); 
    return param + idx*4 ; 
}


char Shape::get_type(unsigned idx) const
{
    assert( idx < num ); 
    return typs[idx] ; 
}
float Shape::get_size(unsigned idx) const
{
    assert( idx < num ); 
    return param[0+4*idx] ; 
}

std::string Shape::desc(unsigned idx) const 
{  
    std::stringstream ss ; 
    ss << " idx: " << idx  ;
    ss << " typ: " << get_type(idx)  ;
    ss << " kludge_outer_aabb: " << kludge_outer_aabb ;  
    ss << " gas_bi_aabb " << gas_bi_aabb ; 
    ss << " param: " ; 
    for(unsigned i=0 ; i < 4 ; i++) ss << param[i+4*idx] << " "  ; 
    ss << " aabb: " ; 
    for(unsigned i=0 ; i < 6 ; i++) ss << aabb[i+6*idx] << " "  ; 
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
    NP::Write(dir.c_str(), "aabb.npy",   aabb,  num, 2, 3 ); 
    NP::Write(dir.c_str(), "param.npy",  param, num, 1, 4 ); 

    assert( node.size() == num ); 
    NP::Write(dir.c_str(), "node.npy",   (float*)node.data(), num, 4, 4 ); 
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






