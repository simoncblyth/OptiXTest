#include "sutil_vec_math.h"

#include "qat4.h"
#include "intersect_node.h"
#include "intersect_tree.h"

#include "Prim.h"
#include "Node.h"

//#include <optix_world.h>
//using namespace optix;

#include <optix_device.h>

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         tmin, , );
rtDeclareVariable(unsigned,      radiance_ray_type, , );
rtDeclareVariable(unsigned,      cameratype, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,      t, rtIntersectionDistance, );

rtBuffer<uchar4, 2>   pixels_buffer;
rtBuffer<float4, 2>   posi_buffer;


static __device__ __inline__ uchar4 make_color(const float3& c)
{
    return make_uchar4( static_cast<unsigned char>(__saturatef(c.x)*255.99f),  
                        static_cast<unsigned char>(__saturatef(c.y)*255.99f),   
                        static_cast<unsigned char>(__saturatef(c.z)*255.99f),   
                        255u);                                                 
}


struct PerRayData
{
    float3   result;
    float4   posi ; 
};

rtDeclareVariable(float3, position,         attribute position, );  
rtDeclareVariable(float3, shading_normal,   attribute shading_normal, );  
rtDeclareVariable(PerRayData, prd, rtPayload, );

rtDeclareVariable(unsigned,  intersect_identity,   attribute intersect_identity, );  
rtDeclareVariable(unsigned, identity,  ,);   
// "identity" is planted into pergi["identity"] 


rtDeclareVariable(rtObject,      top_object, , );

rtBuffer<Prim> prim_buffer;
rtBuffer<Node> node_buffer;
rtBuffer<qat4> itra_buffer;
rtBuffer<float4> plan_buffer;


RT_PROGRAM void raygen()
{
    PerRayData prd;
    prd.result = make_float3( 1.f, 0.f, 0.f ) ; 
    prd.posi = make_float4( 0.f, 0.f, 0.f, 0.f ); 

    const float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;
    const float3 dxyUV = d.x*U + d.y*V ; 
    //                       cameratype     0u perspective           1u orthographic
    const float3 origin    = cameratype == 0u ? eye                    : eye + dxyUV    ; 
    const float3 direction = cameratype == 0u ? normalize( dxyUV + W ) : normalize( W ) ; 

    optix::Ray ray = optix::make_Ray( origin, direction, radiance_ray_type, tmin, RT_DEFAULT_MAX) ; 
    rtTrace(top_object, ray, prd);

    pixels_buffer[launch_index] = make_color( prd.result ) ; 
    posi_buffer[launch_index] = prd.posi ; 
}

RT_PROGRAM void miss()
{
    prd.result = make_float3(1.f, 1.f, 1.f) ;
}

/**
As the primIdx argument is in 0:num_prim-1 need separate prim_buffer per geometry 
unlike nodes and itra where s context level node_buffer and itra_buffer allows 
the pre-7 machinery to more closely match optix7
**/

RT_PROGRAM void intersect(int primIdx)
{
    const Prim* prim = &prim_buffer[primIdx] ;   
    int nodeOffset = prim->nodeOffset() ;  
    int numNode = prim->numNode() ; 
    const Node* node = &node_buffer[nodeOffset] ; 
    const float4* plan = &plan_buffer[0] ;  
    const qat4*   itra = &itra_buffer[0] ;  

    float4 isect ; 
    if(intersect_prim(isect, numNode, node, plan, itra, ray.tmin , ray.origin, ray.direction ))
    {
        if(rtPotentialIntersection(isect.w))
        {
            position = ray.origin + isect.w*ray.direction ;   
            shading_normal = make_float3( isect.x, isect.y, isect.z ); 
            intersect_identity = (( (1u+primIdx) & 0xff ) << 24 ) | ( identity & 0x00ffffff ) ; 
            rtReportIntersection(0);
        }
    }
}



/*
RT_PROGRAM void intersect(int primIdx)
{
    const Node* node = &node_buffer[primIdx] ;   // only working because simple 1:1 Prim:Node
    const float3 center = make_float3( node->q0.f.x, node->q0.f.y, node->q0.f.z) ;
    const float  radius = node->q0.f.w ; 

    const float  t_min = ray.tmin ; 
    const float3 O     = ray.origin - center;
    const float3 D     = ray.direction ; 
 
    float b = dot(O, D);
    float c = dot(O, O)-radius*radius;
    float d = dot(D, D);
    float disc = b*b-d*c;

    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;   // ray has segment within sphere for sdisc > 0.f 
    float root1 = (-b - sdisc)/d ;
    float root2 = (-b + sdisc)/d ;  // root2 > root1 always

    float t_cand = sdisc > 0.f ? ( root1 > t_min ? root1 : root2 ) : t_min ; 
    bool valid_isect = t_cand > t_min ;

    if(valid_isect)
    {
        if(rtPotentialIntersection(t_cand))
        {
            position = ray.origin + t_cand*ray.direction ;   
            shading_normal = ( O + t_cand*D )/radius;
            intersect_identity = (( (1u+primIdx) & 0xff ) << 24 ) | ( identity & 0x00ffffff ) ; 
            rtReportIntersection(0);
        }
    }
}
*/

RT_PROGRAM void bounds (int primIdx, float result[6])
{
    const Prim* prim = &prim_buffer[primIdx] ; 
    const float* aabb = prim->AABB();  

    result[0] = *(aabb+0); 
    result[1] = *(aabb+1); 
    result[2] = *(aabb+2); 
    result[3] = *(aabb+3); 
    result[4] = *(aabb+4); 
    result[5] = *(aabb+5); 

    rtPrintf("// bounds primIdx %d aabb %10.3f %10.3f %10.3f   %10.3f %10.3f %10.3f  \n", primIdx, 
         result[0], result[1], result[2],  
         result[3], result[4], result[5] 
        ); 
}

RT_PROGRAM void closest_hit()
{
    prd.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
    float3 isect = ray.origin + t*ray.direction ;
    prd.posi = make_float4( isect, __uint_as_float(intersect_identity) );
}


