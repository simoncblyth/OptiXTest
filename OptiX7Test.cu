#include <optix.h>

#include "OpticksCSG.h"
#include "Quad.h"
#include "Node.h"
#include "Binding.h"
#include "Params.h"
#include "sutil_vec_math.h"

#include "robust_quadratic_roots.h"

extern "C" { __constant__ Params params ;  }

static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        float3*                normal, 
        float*                 t, 
        float3*                position,
        unsigned*              identity
        )
{
    uint32_t p0, p1, p2, p3 ;
    uint32_t p4, p5, p6, p7 ;

    p0 = float_as_uint( normal->x );
    p1 = float_as_uint( normal->y );
    p2 = float_as_uint( normal->z );
    p3 = float_as_uint( *t );

    p4 = float_as_uint( position->x );
    p5 = float_as_uint( position->y );
    p6 = float_as_uint( position->z );
    p7 = *identity ;
 
    unsigned SBToffset = 0u ; 
    unsigned SBTstride = 1u ; 
    unsigned missSBTIndex = 0u ; 
    const float rayTime = 0.0f ; 

    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            rayTime,
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            SBToffset,
            SBTstride,
            missSBTIndex,
            p0, p1, p2, p3, 
            p4, p5, p6, p7
            );

    normal->x = uint_as_float( p0 );
    normal->y = uint_as_float( p1 );
    normal->z = uint_as_float( p2 );
    *t        = uint_as_float( p3 ); 

    position->x = uint_as_float( p4 );
    position->y = uint_as_float( p5 );
    position->z = uint_as_float( p6 );
    *identity   = p7 ; 
 
}

static __forceinline__ __device__ void setPayload( float3 normal, float t, float3 position, unsigned identity )
{
    optixSetPayload_0( float_as_uint( normal.x ) );
    optixSetPayload_1( float_as_uint( normal.y ) );
    optixSetPayload_2( float_as_uint( normal.z ) );
    optixSetPayload_3( float_as_uint( t ) );

    optixSetPayload_4( float_as_uint( position.x ) );
    optixSetPayload_5( float_as_uint( position.y ) );
    optixSetPayload_6( float_as_uint( position.z ) );
    optixSetPayload_7( identity );
}

__forceinline__ __device__ uchar4 make_color( const float3& normal, unsigned identity )
{
    //float scale = iidx % 2u == 0u ? 0.5f : 1.f ; 
    float scale = 1.f ; 
    return make_uchar4(
            static_cast<uint8_t>( clamp( normal.x, 0.0f, 1.0f ) *255.0f )*scale ,
            static_cast<uint8_t>( clamp( normal.y, 0.0f, 1.0f ) *255.0f )*scale ,
            static_cast<uint8_t>( clamp( normal.z, 0.0f, 1.0f ) *255.0f )*scale ,
            255u
            );
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const float2      d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;


    const unsigned cameratype = params.cameratype ;  
    const float3 dxyUV = d.x * params.U + d.y * params.V ; 
    //                           cameratype 0u:perspective,                    1u:orthographic
    const float3 origin    = cameratype == 0u ? params.eye                     : params.eye + dxyUV    ;
    const float3 direction = cameratype == 0u ? normalize( dxyUV + params.W )  : normalize( params.W ) ;

    float3   normal  = make_float3( 0.5f, 0.5f, 0.5f );
    float    t = 0.f ; 
    float3   position = make_float3( 0.f, 0.f, 0.f );
    unsigned identity = 0u ; 

    trace( 
        params.handle,
        origin,
        direction,
        params.tmin,
        params.tmax,
        &normal, 
        &t, 
        &position,
        &identity
    );

    uchar4 color = make_color( normal, identity );
    const bool yflip = true ; 
    unsigned index = ( yflip ? dim.y - 1 - idx.y : idx.y ) * params.width + idx.x ;

    params.pixels[index] = color ; 
    params.isect[index] = make_float4( position.x, position.y, position.z, uint_as_float(identity)) ; 
}

extern "C" __global__ void __miss__ms()
{
    MissData* ms  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    float3 normal = make_float3( ms->r, ms->g, ms->b );   
    float t_cand = 0.f ; 
    float3 position = make_float3( 0.f, 0.f, 0.f ); 
    unsigned identity = 0u ; 
    setPayload( normal,  t_cand, position, identity );
}



__forceinline__ __device__ 
bool csg_intersect_sphere(const quad& q0, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    float3 center = make_float3(q0.f);
    float radius = q0.f.w;

    float3 O = ray_origin - center;
    float3 D = ray_direction;

    float b = dot(O, D);
    float c = dot(O, O)-radius*radius;
    float d = dot(D, D);

#ifdef CATASTROPHIC_SUBTRACTION_ROOTS
    float disc = b*b-d*c;
    float sdisc = disc > 0.f ? sqrtf(disc) : 0.f ;   // ray has segment within sphere for sdisc > 0.f 
    float root1 = (-b - sdisc)/d ;
    float root2 = (-b + sdisc)/d ;  // root2 > root1 always
#else
    float root1, root2, disc, sdisc ;   
    robust_quadratic_roots(root1, root2, disc, sdisc, d, b, c ) ; //  Solving:  d t^2 + 2 b t +  c = 0    root2 > root1 
#endif

    float t_cand = sdisc > 0.f ? ( root1 > t_min ? root1 : root2 ) : t_min ;

    bool valid_isect = t_cand > t_min ;
    if(valid_isect)
    {
        isect.x = (O.x + t_cand*D.x)/radius ;   // normalized by construction
        isect.y = (O.y + t_cand*D.y)/radius ;
        isect.z = (O.z + t_cand*D.z)/radius ;
        isect.w = t_cand ;
    }
    return valid_isect ;
}


__forceinline__ __device__ 
bool csg_intersect_zsphere(const quad& q0, const quad& q1, const float& t_min, float4& isect, const float3& ray_origin, const float3& ray_direction )
{
    const float3 center = make_float3(q0.f);
    float3 O = ray_origin - center;  
    float3 D = ray_direction;
    const float radius = q0.f.w;

    float b = dot(O, D);               // t of closest approach to sphere center
    float c = dot(O, O)-radius*radius; // < 0. indicates ray_origin inside sphere
    if( c > 0.f && b > 0.f ) return false ;    

    // Cannot intersect when ray origin outside sphere and direction away from sphere.
    // Whether early exit speeds things up is another question ... 

    const bool QCAP = true ; 
    const bool PCAP = true ;  

    const float2 zdelta = make_float2(q1.f);
    const float zmax = center.z + zdelta.y ; 
    const float zmin = center.z + zdelta.x ;    

    float d = dot(D, D);               // NB NOT assuming normalized ray_direction

    float t1sph, t2sph, disc, sdisc ;    
    robust_quadratic_roots(t1sph, t2sph, disc, sdisc, d, b, c); //  Solving:  d t^2 + 2 b t +  c = 0 

    float z1sph = ray_origin.z + t1sph*ray_direction.z ;  // sphere z intersects
    float z2sph = ray_origin.z + t2sph*ray_direction.z ; 

    float idz = 1.f/ray_direction.z ; 
    float t_QCAP = QCAP ? (zmax - ray_origin.z)*idz : t_min ;   // cap intersects,  t_min for cap not enabled
    float t_PCAP = PCAP ? (zmin - ray_origin.z)*idz : t_min ;

    float t1cap = fminf( t_QCAP, t_PCAP ) ;   // order cap intersects along the ray 
    float t2cap = fmaxf( t_QCAP, t_PCAP ) ;   // t2cap > t1cap 

    // disqualify plane intersects outside sphere t range
    if(t1cap < t1sph || t1cap > t2sph) t1cap = t_min ; 
    if(t2cap < t1sph || t2cap > t2sph) t2cap = t_min ; 

    // hmm somehow is seems unclean to have to use both z and t language

    float t_cand = t_min ; 
    if(sdisc > 0.f)
    {
        if(      t1sph > t_min && z1sph > zmin && z1sph < zmax )  t_cand = t1sph ;  // t1sph qualified and t1cap disabled or disqualified -> t1sph
        else if( t1cap > t_min )                                  t_cand = t1cap ;  // t1cap qualifies -> t1cap 
        else if( t2cap > t_min )                                  t_cand = t2cap ;  // t2cap qualifies -> t2cap
        else if( t2sph > t_min && z2sph > zmin && z2sph < zmax)   t_cand = t2sph ;  // t2sph qualifies and t2cap disabled or disqialified -> t2sph

        //rtPrintf("csg_intersect_zsphere t_min %7.3f t1sph %7.3f t1cap %7.3f t2cap %7.3f t2sph %7.3f t_cand %7.3f \n", t_min, t1sph, t1cap, t2cap, t2sph, t_cand ); 
    }

    bool valid_isect = t_cand > t_min ;

    if(valid_isect)
    {
        isect.w = t_cand ;
        if( t_cand == t1sph || t_cand == t2sph)
        {
            isect.x = (O.x + t_cand*D.x)/radius ; // normalized by construction
            isect.y = (O.y + t_cand*D.y)/radius ;
            isect.z = (O.z + t_cand*D.z)/radius ;
        }
        else
        {
            isect.x = 0.f ;
            isect.y = 0.f ;
            isect.z = t_cand == t_PCAP ? -1.f : 1.f ;
        }
    }
    return valid_isect ;
}








extern "C" __global__ void __intersection__is()
{
    HitGroupData* hg  = reinterpret_cast<HitGroupData*>( optixGetSbtDataPointer() );

    const Node* node = hg->node ;
    const unsigned typecode = node->typecode() ;  

    const float3 orig = optixGetObjectRayOrigin();
    const float3 dir  = optixGetObjectRayDirection();
    const float  t_min = optixGetRayTmin() ; 

    float4 isect ; 
    bool valid_isect ; 
    switch(typecode)
    {
        case CSG_SPHERE:  valid_isect = csg_intersect_sphere(  node->q0,           t_min, isect, orig, dir ) ;  break ; 
        case CSG_ZSPHERE: valid_isect = csg_intersect_zsphere( node->q0, node->q1, t_min, isect, orig, dir ) ;  break ; 
    }

    float t_cand = isect.w ; 

    if(valid_isect)
    {
        const float3 position = orig + t_cand*dir ;     // TODO: can this be done in CH ?
        const unsigned hitKind = 0u ;  // user hit kind

        unsigned a0, a1, a2, a3;   // attribute registers
        unsigned a4, a5, a6, a7;

        a0 = float_as_uint( isect.x );
        a1 = float_as_uint( isect.y );
        a2 = float_as_uint( isect.z );
        a3 = float_as_uint( isect.w ) ; 

        a4 = float_as_uint( position.x );
        a5 = float_as_uint( position.y );
        a6 = float_as_uint( position.z );
        a7 = 0u ; 

        optixReportIntersection(
                isect.w,      
                hitKind,       
                a0, a1, a2, a3, 
                a4, a5, a6, a7
                );
    }
}

extern "C" __global__ void __closesthit__ch()
{
    const float3 shading_normal =
        make_float3(
                uint_as_float( optixGetAttribute_0() ),
                uint_as_float( optixGetAttribute_1() ),
                uint_as_float( optixGetAttribute_2() )
                );

    float t = int_as_float( optixGetAttribute_3() ) ; 

    const float3 position =
        make_float3(
                uint_as_float( optixGetAttribute_4() ),
                uint_as_float( optixGetAttribute_5() ),
                uint_as_float( optixGetAttribute_6() )
                );

    //unsigned bindex = optixGetAttribute_7() ;
    //unsigned instanceIndex = optixGetInstanceIndex() ;    

    unsigned instance_id = optixGetInstanceId() ;        // see IAS_Builder::Build and InstanceId.h 
    unsigned prim_id  = 1u + optixGetPrimitiveIndex() ;  // see GAS_Builder::MakeCustomPrimitivesBI 
    unsigned identity = (( prim_id & 0xff ) << 24 ) | ( instance_id & 0x00ffffff ) ; 
 
    float3 normal = normalize( optixTransformNormalFromObjectToWorldSpace( shading_normal ) ) * 0.5f + 0.5f ;  

    const float3 world_origin = optixGetWorldRayOrigin() ; 
    const float3 world_direction = optixGetWorldRayDirection() ; 
    const float3 world_position = world_origin + t*world_direction ; 

    setPayload( normal, t,  world_position, identity );
}

