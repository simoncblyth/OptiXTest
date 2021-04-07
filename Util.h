#pragma once

#include "sutil_vec_math.h"
#include <vector>
#include <array>
#include <glm/glm.hpp>

struct Util
{
    static const char* PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext=".cu" );

    static void ParseGridSpec(       std::array<int,9>& grid, const char* spec ) ;
    static void GridMinMax(    const std::array<int,9>& grid, int& mn, int& mx ) ;
    static void GridMinMax    (const std::array<int,9>& grid, int3&mn, int3& mx) ;

    static void DumpGrid(      const std::array<int,9>& grid ) ;

    static unsigned Encode4(const char* s); 


    template <typename T>
    static T ato_( const char* a );

    template <typename T>
    static T GetEValue(const char* key, T fallback);  

    template <typename T>
    static void GetEVector(std::vector<T>& vec, const char* key, const char* fallback );

    static void GetEVec(glm::vec3& v, const char* key, const char* fallback );

    static void GetEVec(glm::vec4& v, const char* key, const char* fallback );

    template <typename T>
    static std::string Present(std::vector<T>& vec);

    static bool StartsWith( const char* s, const char* q);


};
