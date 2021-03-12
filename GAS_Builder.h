#pragma once

#include <vector>
#include "GAS.h"
#include "BI.h"

struct Shape ; 

struct GAS_Builder
{
    static void Build(GAS& gas, const Shape* sh  ); 

    // original approach GAS:BI:AABB  1:N:N   : one BI for every layer of the compound GAS

    static void                Build_1NN(GAS& gas, const Shape* sh  ); 
    static BI MakeCustomPrimitivesBI_1NN(const Shape* sh, unsigned i ); 

    // 2nd try approach  GAS:BI:AABB  1:1:N  : only one BI for all layers of compound GAS

    static void                Build_11N(GAS& gas, const Shape* sh  ); 
    static BI MakeCustomPrimitivesBI_11N(const Shape* sh); 

    static void Build(GAS& gas);
};


