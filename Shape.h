#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "Node.h"
#include "AABB.h"


/**
Shape
========

Hmm: Solid from Foundry is superceeding this 

**/

struct Shape
{
    enum { GBA_1NN=0, GBA_11N=1 } ; 
    static unsigned Type(char typ);

    static constexpr unsigned prim_size = 4 ; 
    static constexpr unsigned node_size = 4*4 ; 

    unsigned num ; 
    char*  typs ; 

    int kludge_outer_aabb ; 
    int gas_bi_aabb ; 

    bool is1NN() const ; 
    bool is11N() const ; 


    Shape(const char typ, float sz);
    Shape(const char* typs, const std::vector<float>& szs);
    virtual ~Shape();
    void init(const std::vector<float>& szs );

    const float* get_aabb(unsigned idx) const ;
    unsigned get_aabb_stride() const ;  // in bytes 

    const Node* get_node(unsigned idx) const ;

    int*                 get_prim(unsigned idx) const ;
    const    glm::ivec4& get_prim_(unsigned idx) const ; 
    unsigned             get_num_node(unsigned idx) const ;

    void add_sphere(float radius);
    void add_zsphere(float radius);
    void add_prim(int num_node);
    char  get_type(unsigned idx) const ;



    std::string desc(unsigned idx) const ;
    std::string desc() const ;

    void write(const char* base, const char* rel, unsigned idx) const ;

    static void Dump(const float* f, const int ni, const char* label);
    static void Dump(const glm::mat4& nd );
    static void Dump(const std::vector<glm::mat4>& nds);

    std::vector<glm::ivec4> prim ; 
    std::vector<AABB>       aabb ; 
    std::vector<Node>       node ; 

};



