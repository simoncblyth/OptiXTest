// name=qat4 ; nvcc $name.cu -ccbin /usr/bin/clang -o /tmp/$name && /tmp/$name

#include "cuda.h"

#include "sutil_vec_math.h"
#include "qat4.h"
#include "stdio.h"

#include <vector>
#include <cassert>


__global__ void test_multiply(int threads_per_launch, int thread_offset, const qat4* q_arr, const float4* v_arr, float4* vq_arr, float4* qv_arr )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    
    const qat4& q = q_arr[id] ; 
    const float4& v = v_arr[id] ;  

    vq_arr[id] = v * q ; 
    qv_arr[id] = q * v ; 

    //vq_arr[id] = v ; 
    //qv_arr[id] = v ; 
}


void check( const float& a, const float& b )
{
    assert( a == b ); 
}

void check( const float4& a , const float4& b )
{
    check(a.x, b.x); 
    check(a.y, b.y); 
    check(a.z, b.z); 
    check(a.w, b.w); 
}


int main(int argc, char** argv)
{
    printf("%s\n", argv[0]); 

    qat4 m ; 
    m.q3.f.x = 1000.f ; 
    m.q3.f.y = 1000.f ; 
    m.q3.f.z = 1000.f ; 

    int ni = 10 ; 
    std::vector<qat4>   q(ni) ; 
    std::vector<float4> v(ni) ; 
    std::vector<float4> vq(ni) ; 
    std::vector<float4> qv(ni) ; 

    for(int i=0 ; i < ni ; i++) 
    {
        q[i] = m ; 
        v[i] = make_float4(float(i+10), float(i+20), float(i+30), 1.f); 
    }

    for(int i=0 ; i < ni ; i++) 
    {
        const float4& v0 = v[i] ;  
        const qat4&   q0 = q[i] ;  
        printf(" v0 : %10.4f %10.4f %10.4f %10.4f \n", v0.x, v0.y, v0.z, v0.w ); 
    }


    CUdeviceptr d_q ;
    cudaMalloc((void**)&d_q, sizeof(qat4)*ni );
    cudaMemcpy( (void*)d_q, q.data(), sizeof(qat4)*ni, cudaMemcpyHostToDevice );

    CUdeviceptr d_v ;
    cudaMalloc((void**)&d_v, sizeof(float4)*ni );
    cudaMemcpy( (void*)d_v, v.data(), sizeof(float4)*ni, cudaMemcpyHostToDevice );

    CUdeviceptr d_vq ;
    cudaMalloc((void**)&d_vq, sizeof(float4)*ni );
    cudaMemset((void*)d_vq, 0, sizeof(float4)*ni );

    CUdeviceptr d_qv ;
    cudaMalloc((void**)&d_qv, sizeof(float4)*ni );
    cudaMemset((void*)d_qv, 0, sizeof(float4)*ni );


    int blocks_per_launch = 1 ;
    int threads_per_launch = ni ;
    int threads_per_block = threads_per_launch ;
    int thread_offset = 0 ;  

    test_multiply<<<blocks_per_launch,threads_per_block>>>(threads_per_launch, thread_offset, (const qat4*)d_q, (const float4*)d_v, (float4*)d_vq, (float4*)d_qv );    

    cudaMemcpy( vq.data(), (void*)d_vq, sizeof(float4)*ni, cudaMemcpyDeviceToHost );
    cudaMemcpy( qv.data(), (void*)d_qv, sizeof(float4)*ni, cudaMemcpyDeviceToHost );

    for(int i=0 ; i < ni ; i++)
    {
         const float4& vi = v[i] ;  
         const qat4&   qi = q[i] ;  

         float4 vq0 = vi * qi ;
         float4 qv0 = qi * vi ;
 
         const float4& vq1 = vq[i] ;  
         const float4& qv1 = qv[i] ;  

         check( vq0, vq1 ); 
         check( qv0, qv1 ); 

         //printf(" vq0 : %10.4f %10.4f %10.4f %10.4f    vq1: %10.4f %10.4f %10.4f %10.4f \n", vq0.x, vq0.y, vq0.z, vq0.w, vq1.x, vq1.y, vq1.z, vq1.w  ); 
         printf(" qv0 : %10.4f %10.4f %10.4f %10.4f    qv1: %10.4f %10.4f %10.4f %10.4f \n", qv0.x, qv0.y, qv0.z, qv0.w, qv1.x, qv1.y, qv1.z, qv1.w  ); 
    }

    cudaDeviceSynchronize();

    return 0 ; 
}
