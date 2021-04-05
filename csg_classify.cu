// name=csg_classify ; nvcc $name.cu -ccbin=/usr/bin/clang -std=c++11 -DDEBUG=1 -lstdc++ -o /tmp/$name && /tmp/$name

#ifdef DEBUG
#include <stdio.h>
#endif

#include "OpticksCSG.h"
#include "csg_classify.h"

__global__ void test_lut(int threads_per_launch, int thread_offset )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    
    LUT lut ;  

    bool ACloser = true ; 
    OpticksCSG_t operation = CSG_UNION ; 
    IntersectionState_t stateA = State_Exit ; 
    IntersectionState_t stateB = State_Exit ; 
    int action = lut.lookup( operation, stateA, stateB, ACloser );
#ifdef DEBUG
    printf("//csg_classify.cu:test_lut  action %d \n", action ); 
#endif
}

int main(int argc, char** argv)
{

    int blocks_per_launch = 1 ; 
    int threads_per_block = 1 ; 
    int threads_per_launch = 1 ; 
    int thread_offset = 0 ; 

    test_lut<<<blocks_per_launch,threads_per_block>>>(threads_per_launch, thread_offset );    

    cudaDeviceSynchronize();

    return 0 ; 
}


