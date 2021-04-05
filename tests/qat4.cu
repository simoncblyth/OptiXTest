// name=qat4 ; nvcc $name.cu -ccbin /usr/bin/clang -o /tmp/$name && /tmp/$name

#include "cuda.h"

#include "sutil_vec_math.h"
#include "qat4.h"
#include "stdio.h"

#include <vector>
#include <cassert>


__global__ void test_multiply(int threads_per_launch, int thread_offset, const qat4* q_arr, const float3* v_arr, float3* vq_arr, float3* qv_arr )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= threads_per_launch) return;
    
    const qat4& q = q_arr[id] ; 
    const float3& v = v_arr[id] ;  

    vq_arr[id] = q.left_multiply(v, 1.f) ; 
    qv_arr[id] = q.right_multiply(v, 1.f)  ; 

    //vq_arr[id] = v ; 
    //qv_arr[id] = v ; 
}


void check( const float& a, const float& b )
{
    assert( a == b ); 
}
void check( const float3& a , const float3& b )
{
    check(a.x, b.x); 
    check(a.y, b.y); 
    check(a.z, b.z); 
}
void check( const float4& a , const float4& b )
{
    check(a.x, b.x); 
    check(a.y, b.y); 
    check(a.z, b.z); 
    check(a.w, b.w); 
}


void print( const char* label,  const float4& f )
{
    printf(" %s  (%10.4f, %10.4f, %10.4f, %10.4f) \n", label, f.x, f.y, f.z, f.w );
}
void print( const char* label, const qat4& q )
{
    printf("%s \n", label); 
    print("q.q0", q.q0.f ); 
    print("q.q1", q.q1.f ); 
    print("q.q2", q.q2.f ); 
    print("q.q3", q.q3.f ); 
}


int main(int argc, char** argv)
{
    printf("%s\n", argv[0]); 

    qat4 m ; 
    m.q0.f.x = 1.f ; 
    m.q1.f.y = 1.f ; 
    m.q2.f.z = 1.f ; 
    m.q3.f.w = 1.f ; 

    m.q3.f.x = 1000.f ; 
    m.q3.f.y = 1000.f ; 
    m.q3.f.z = 1000.f ; 
    print("m", m); 


    int ni = 10 ; 
    std::vector<qat4>   q(ni) ; 
    std::vector<float3> v(ni) ; 
    std::vector<float3> vq(ni) ; 
    std::vector<float3> qv(ni) ; 

    for(int i=0 ; i < ni ; i++) 
    {
        q[i] = m ; 
        v[i] = make_float3(float(i+10), float(i+20), float(i+30) ); 
    }

    for(int i=0 ; i < ni ; i++) 
    {
        const float3& v0 = v[i] ;  
        const qat4&   q0 = q[i] ;  
        printf(" v0 : %10.4f %10.4f %10.4f  \n", v0.x, v0.y, v0.z  ); 
    }


    CUdeviceptr d_q ;
    cudaMalloc((void**)&d_q, sizeof(qat4)*ni );
    cudaMemcpy( (void*)d_q, q.data(), sizeof(qat4)*ni, cudaMemcpyHostToDevice );

    CUdeviceptr d_v ;
    cudaMalloc((void**)&d_v, sizeof(float3)*ni );
    cudaMemcpy( (void*)d_v, v.data(), sizeof(float3)*ni, cudaMemcpyHostToDevice );

    CUdeviceptr d_vq ;
    cudaMalloc((void**)&d_vq, sizeof(float3)*ni );
    cudaMemset((void*)d_vq, 0, sizeof(float3)*ni );

    CUdeviceptr d_qv ;
    cudaMalloc((void**)&d_qv, sizeof(float3)*ni );
    cudaMemset((void*)d_qv, 0, sizeof(float3)*ni );


    int blocks_per_launch = 1 ;
    int threads_per_launch = ni ;
    int threads_per_block = threads_per_launch ;
    int thread_offset = 0 ;  

    test_multiply<<<blocks_per_launch,threads_per_block>>>(threads_per_launch, thread_offset, (const qat4*)d_q, (const float3*)d_v, (float3*)d_vq, (float3*)d_qv );    

    cudaMemcpy( vq.data(), (void*)d_vq, sizeof(float3)*ni, cudaMemcpyDeviceToHost );
    cudaMemcpy( qv.data(), (void*)d_qv, sizeof(float3)*ni, cudaMemcpyDeviceToHost );

    for(int i=0 ; i < ni ; i++)
    {
         const float3& vi = v[i] ;  
         const qat4&   qi = q[i] ;  

         float3 vq0 = qi.left_multiply(vi, 1.f) ;
         float3 qv0 = qi.right_multiply(vi, 1.f) ;
 
         const float3& vq1 = vq[i] ;  
         const float3& qv1 = qv[i] ;  

         check( vq0, vq1 ); 
         check( qv0, qv1 ); 

         printf(" qv0 : %10.4f %10.4f %10.4f      qv1: %10.4f %10.4f %10.4f   \n", qv0.x, qv0.y, qv0.z,     qv1.x, qv1.y, qv1.z  ); 
    }

    cudaDeviceSynchronize();

    return 0 ; 
}
