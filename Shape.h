#pragma once
#include <vector>
#include <string>

struct Shape
{
    enum { GBA_1NN=0, GBA_11N=1 } ; 
    enum { ZERO=0, SPHERE=1, BOX=2 };
    static unsigned Type(char typ);

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
    char  get_type(unsigned idx) const ;
    float get_size(unsigned idx) const ;

    std::string desc(unsigned idx) const ;
    std::string desc() const ;

    void write(const char* base, const char* rel, unsigned idx) const ;

};



