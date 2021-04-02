cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name OptiXTest)
project(${name})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(CUDA REQUIRED)   
find_package(OptiX REQUIRED)   

message(STATUS "OptiX_FOUND   : ${OptiX_FOUND} ")
message(STATUS "OptiX_VERSION : ${OptiX_VERSION} ")
message(STATUS "OptiX_INCLUDE : ${OptiX_INCLUDE} ")

set(OptiX_VERSION_ 0)
if(OptiX_INCLUDE)
   file(READ "${OptiX_INCLUDE}/optix.h" _contents)
   string(REGEX REPLACE "\n" ";" _contents "${_contents}")
   foreach(_line ${_contents})
        if (_line MATCHES "#define OPTIX_VERSION ([0-9]+)")
            set(OptiX_VERSION_ ${CMAKE_MATCH_1} )
        endif()
   endforeach()
endif()

message(STATUS "OptiX_VERSION_ : ${OptiX_VERSION_} ")
message(STATUS "writing ${CMAKE_BINARY_DIR}/buildenv.sh " )

FILE(WRITE ${CMAKE_BINARY_DIR}/buildenv.sh 
"export OPTIX_VERSION=${OptiX_VERSION_}\n"
)


add_library(ImpOptiX INTERFACE IMPORTED) 
set_target_properties(ImpOptiX PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${OptiX_INCLUDE}")
if(OptiX_VERSION_ LESS 70000)
set_target_properties(ImpOptiX PROPERTIES INTERFACE_LINK_LIBRARIES "optix;optixu;optix_prime")
endif()


include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${OptiX_INCLUDE}
    ${CMAKE_INSTALL_PREFIX}/externals/glm/glm
    ${CMAKE_CURRENT_SOURCE_DIR} 
)


set(COMPUTE_CAPABILITY 30)

set(CUDA_NVCC_FLAGS)
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
list(APPEND CUDA_NVCC_FLAGS "-O2")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")


set(SOURCES)
set(CU_SOURCES)


list(APPEND SOURCES
    OptiXTest.cc
    Sys.cc

    Solid.cc
    Prim.cc
    PrimSpec.cc
    Node.cc

    Grid.cc
    Geo.cc
    Util.cc 
    Frame.cc 
    Params.cc 
    Foundry.cc
    CU.cc
)

if(OptiX_VERSION_ GREATER_EQUAL 70000)
   list(APPEND CU_SOURCES OptiX7Test.cu)

   list(APPEND SOURCES  
        Ctx.cc 
        Properties.cc 
        GAS.cc 
        GAS_Builder.cc 
        PIP.cc 
        IAS.cc 
        IAS_Builder.cc 
        SBT.cc 
    )
else()

   list(APPEND SOURCES  
        Six.cc 
   )

   list(APPEND CU_SOURCES OptiX6Test.cu)
endif()


CUDA_WRAP_SRCS( ${name} PTX _generated_PTX_files ${CU_SOURCES} )
CUDA_WRAP_SRCS( ${name} OBJ _generated_OBJ_files ${SOURCES} )

message(STATUS "_generated_OBJ_files ${_generated_OBJ_files}") 
message(STATUS "_generated_PTX_files ${_generated_PTX_files}") 


add_executable( ${name} ${_generated_OBJ_files} ${_generated_PTX_files} ${SOURCES} )


target_link_libraries(${name} 
     PUBLIC 
         ${CUDA_LIBRARIES}
         ImpOptiX
)
install(TARGETS ${name}               DESTINATION bin)
install(FILES ${_generated_PTX_files} DESTINATION ptx)

