#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "Node.h"


struct Shape
{
    enum { GBA_1NN=0, GBA_11N=1 } ; 
    enum { ZERO=0, SPHERE=1, BOX=2 };
    static unsigned Type(char typ);

    static constexpr unsigned prim_size = 4 ; 
    static constexpr unsigned node_size = 4*4 ; 
    static constexpr unsigned tran_size = 4*4 ;  // ?3*4*4
    static constexpr unsigned plan_size = 4 ; 

    unsigned num ; 
    char*  typs ; 
    float* param ;  // 4*num
    float* aabb ;   // 6*num 

    int kludge_outer_aabb ; 
    int gas_bi_aabb ; 

    bool is1NN() const ; 
    bool is11N() const ; 

    Shape(const char typ, float sz);
    Shape(const char* typs, const std::vector<float>& szs);
    virtual ~Shape();
    void init(const std::vector<float>& szs );

    float* get_aabb(unsigned idx) const ;
    float* get_param(unsigned idx) const ;

    const Node* get_node(unsigned idx) const ;
    int*   get_prim(unsigned idx) const ;
    const glm::ivec4& get_prim_(unsigned idx) const ; 
    unsigned get_num_node(unsigned idx) const ;

    void add_prim(int num_node);

    char  get_type(unsigned idx) const ;
    float get_size(unsigned idx) const ;

    std::string desc(unsigned idx) const ;
    std::string desc() const ;

    void write(const char* base, const char* rel, unsigned idx) const ;

    static void Dump(const float* f, const int ni, const char* label);
    static void Dump(const glm::mat4& nd );
    static void Dump(const std::vector<glm::mat4>& nds);

    std::vector<glm::ivec4> prim ; 
    std::vector<Node>       node ; 
    std::vector<glm::mat4>  tran ; 
    std::vector<glm::vec4>  plan ; 

};



