
#include <optix_world.h>
using namespace optix;

rtDeclareVariable(float3,        eye, , );
rtDeclareVariable(float3,        U, , );
rtDeclareVariable(float3,        V, , );
rtDeclareVariable(float3,        W, , );
rtDeclareVariable(float,         tmin, , );
rtDeclareVariable(unsigned,     radiance_ray_type, , );

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint2, launch_dim,   rtLaunchDim, );

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float,      t, rtIntersectionDistance, );

rtBuffer<uchar4, 2>   pixels_buffer;
rtBuffer<float4, 2>   posi_buffer;


static __device__ __inline__ optix::uchar4 make_color(const optix::float3& c)
{
    return optix::make_uchar4( static_cast<unsigned char>(__saturatef(c.x)*255.99f),  
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

rtBuffer<float4> shape_buffer;


RT_PROGRAM void raygen()
{
    PerRayData prd;
    prd.result = make_float3( 1.f, 0.f, 0.f ) ; 
    prd.posi = make_float4( 0.f, 0.f, 0.f, 0.f ); 

    float2 d = make_float2(launch_index) / make_float2(launch_dim) * 2.f - 1.f ;

    optix::Ray ray = optix::make_Ray( eye, normalize(d.x*U + d.y*V + W), radiance_ray_type, tmin, RT_DEFAULT_MAX) ; 
    rtTrace(top_object, ray, prd);

    pixels_buffer[launch_index] = make_color( prd.result ) ; 
    posi_buffer[launch_index] = prd.posi ; 
}

RT_PROGRAM void miss()
{
    prd.result = make_float3(1.f, 1.f, 1.f) ;
}

RT_PROGRAM void intersect(int primIdx)
{
    const float4& shape = shape_buffer[primIdx] ; 
    const float  radius = shape.w ; 
    const float  t_min = ray.tmin ; 

    const float3 center = make_float3( shape.x, shape.y, shape.z) ;
    const float3 O      = ray.origin - center;
    const float3 D      = ray.direction ; 
 
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

RT_PROGRAM void bounds (int primIdx, float result[6])
{
    const float4& shape = shape_buffer[primIdx] ; 
    const float  radius = shape.w ; 
    rtPrintf("// bounds primIdx %d radius %10.3f \n", primIdx, radius ); 

    optix::Aabb* aabb = (optix::Aabb*)result;
    float3 mn = make_float3( -radius, -radius, -radius ); 
    float3 mx = make_float3(  radius,  radius,  radius ); 
    aabb->set(mn, mx);
}

RT_PROGRAM void closest_hit()
{
    prd.result = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal))*0.5f + 0.5f;
    float3 isect = ray.origin + t*ray.direction ;
    prd.posi = make_float4( isect, __uint_as_float(intersect_identity) );
}



