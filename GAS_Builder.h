#pragma once

#include <vector>
#include "GAS.h"
#include "Prim.h"
#include "BI.h"

//struct Shape ; 

/**
GAS_Builder
=============

* original approach GAS:BI:AABB  1:N:N   : one BI for every layer of the compound GAS

* 2nd try approach  GAS:BI:AABB  1:1:N  : only one BI for all layers of compound GAS

**/


struct GAS_Builder
{
    //static void Build(     GAS& gas, const float* aabb_base, unsigned num_aabb, unsigned stride_in_bytes  );
    //static BI MakeCustomPrimitivesBI_11N( const float* aabb, unsigned num_aabb, unsigned stride_in_bytes ) ;
    //static void Build_11N( GAS& gas, const float* aabb_base, unsigned num_aabb, unsigned stride_in_bytes );


    static void Build(     GAS& gas, const PrimSpec& psd );
    static void Build_11N( GAS& gas, const PrimSpec& psd );
    static BI MakeCustomPrimitivesBI_11N(const PrimSpec& psd);


    static void Build_1NN( GAS& gas, const float* aabb_base, unsigned num_aabb, unsigned stride_in_bytes  );
    static BI MakeCustomPrimitivesBI_1NN( const float* aabb, unsigned num_aabb, unsigned stride_in_bytes, unsigned primitiveIndexOffset ) ; 


    static void DumpAABB(                const float* aabb, unsigned num_aabb, unsigned stride_in_bytes ) ; 

    //static void Build(    GAS& gas, const Shape* sh  ); 
    //static BI MakeCustomPrimitivesBI_1NN(const Shape* sh, unsigned i ); 
    //static BI MakeCustomPrimitivesBI_11N(const Shape* sh); 

    static void BoilerPlate(GAS& gas);  
};


