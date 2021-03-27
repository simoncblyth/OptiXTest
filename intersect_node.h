#pragma once

#if defined(__CUDACC__) || defined(__CUDABE__)
#    define INTERSECT_FUNC __forceinline__ __device__
#else
#    define INTERSECT_FUNC
#endif


#define RT_DEFAULT_MAX 1.e27f

#if defined(__CUDACC__)
#include "math_constants.h"
#else

union uif_t 
{
    unsigned u ; 
    int i ; 
    float f ; 
};

float __int_as_float(int i)
{
    uif_t uif ; 
    uif.i = i ; 
    return uif.f ; 
}

#define CUDART_INF_F            __int_as_float(0x7f800000)

#endif


#include "OpticksCSG.h"
#include "Quad.h"
#include "Node.h"
#include "Prim.h"
#include "robust_quadratic_roots.h"


INTERSECT_FUNC
bool intersect_node_sphere(float4& isect, const quad& q0, const float& t_min, const float3& ray_origin, const float3& ray_direction )
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


INTERSECT_FUNC
bool intersect_node_zsphere(float4& isect, const quad& q0, const quad& q1, const float& t_min, const float3& ray_origin, const float3& ray_direction )
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


INTERSECT_FUNC
bool intersect_node_convexpolyhedron( float4& isect, const Prim* prim, const Node* node, const float4* plan, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    float t0 = -CUDART_INF_F ; 
    float t1 =  CUDART_INF_F ; 

    float3 t0_normal = make_float3(0.f);
    float3 t1_normal = make_float3(0.f);

    unsigned num_plan = 6 ;  // TODO : get this from Prim 

    for(unsigned i=0 ; i < num_plan ; i++) 
    {    
        float4 plane = plan[i];    // TODO: may need offsets here 
        float3 n = make_float3(plane);
        float dplane = plane.w ;

         // RTCD p199,  
         //            n.X = dplane
         //   
         //             n.(o+td) = dplane
         //            no + t nd = dplane
         //                    t = (dplane - no)/nd
         //   

        float nd = dot(n, ray_direction); // -ve: entering, +ve exiting halfspace  
        float no = dot(n, ray_origin ) ;  //  distance from coordinate origin to ray origin in direction of plane normal 
        float dist = no - dplane ;        //  subtract plane distance from origin to get signed distance from plane, -ve inside 
        float t_cand = -dist/nd ;

        bool parallel_inside = nd == 0.f && dist < 0.f ;   // ray parallel to plane and inside halfspace
        bool parallel_outside = nd == 0.f && dist > 0.f ;  // ray parallel to plane and outside halfspac

        if(parallel_inside) continue ;       // continue to next plane 
        if(parallel_outside) return false ;  // <-- without early exit, this still works due to infinity handling 

        //    NB ray parallel to plane and outside halfspace 
        //         ->  t_cand = -inf 
        //                 nd = 0.f 
        //                t1 -> -inf  

        if( nd < 0.f)  // entering 
        {
            if(t_cand > t0)
            {
                t0 = t_cand ;
                t0_normal = n ;
            }
        }
        else     // exiting
        {
            if(t_cand < t1)
            {
                t1 = t_cand ;
                t1_normal = n ;
            }
        }
    }

    bool valid_intersect = t0 < t1 ;
    if(valid_intersect)
    {
        if( t0 > t_min )
        {
            isect.x = t0_normal.x ;
            isect.y = t0_normal.y ;
            isect.z = t0_normal.z ;
            isect.w = t0 ;
        }
        else if( t1 > t_min )
        {
            isect.x = t1_normal.x ;
            isect.y = t1_normal.y ;
            isect.z = t1_normal.z ;
            isect.w = t1 ;
        }
    }
    return valid_intersect ;
}


/**
intersect_node_cone
=====================






**/

INTERSECT_FUNC
bool intersect_node_cone( float4& isect, const quad& q0, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    float r1 = q0.f.x ; 
    float z1 = q0.f.y ; 
    float r2 = q0.f.z ; 
    float z2 = q0.f.w ;   // z2 > z1

    float tth = (r2-r1)/(z2-z1) ;
    float tth2 = tth*tth ; 
    float z0 = (z2*r1-z1*r2)/(r1-r2) ;  // apex

#ifdef DEBUG
    printf(" r1 %10.4f z1 %10.4f r2 %10.4f z2 %10.4f : z0 %10.4f \n", r1, z1, r2, z2, z0 );  
#endif
 
    float r1r1 = r1*r1 ; 
    float r2r2 = r2*r2 ; 

    const float3& o = ray_origin ;
    const float3& d = ray_direction ;

    //  cone with apex at [0,0,z0]  and   r1/(z1-z0) = tanth  for any r1,z1 on the cone
    //
    //     x^2 + y^2  - (z - z0)^2 tanth^2 = 0 
    //     x^2 + y^2  - (z^2 -2z0 z - z0^2) tanth^2 = 0 
    //
    //   Gradient:    [2x, 2y, (-2z tanth^2) + 2z0 tanth^2 ] 
    //
    //   (o.x+ t d.x)^2 + (o.y + t d.y)^2 - (o.z - z0 + t d.z)^2 tth2 = 0 
    // 
    // quadratic in t :    c2 t^2 + 2 c1 t + c0 = 0 

    float c2 = d.x*d.x + d.y*d.y - d.z*d.z*tth2 ;
    float c1 = o.x*d.x + o.y*d.y - (o.z-z0)*d.z*tth2 ; 
    float c0 = o.x*o.x + o.y*o.y - (o.z-z0)*(o.z-z0)*tth2 ;
    float disc = c1*c1 - c0*c2 ; 

#ifdef DEBUG
    printf(" c2 %10.4f c1 %10.4f c0 %10.4f disc %10.4f : tth %10.4f \n", c2, c1, c0, disc, tth  );  
#endif
 


    // * cap intersects (including axial ones) will always have potentially out of z-range cone intersects 
    // * cone intersects will have out of r-range plane intersects, other than rays within xy plane
 
    bool valid_isect = false ;
 
    if(disc > 0.f)  // has intersects with infinite cone
    {
        float sdisc = sqrtf(disc) ;   
        float root1 = (-c1 - sdisc)/c2 ;
        float root2 = (-c1 + sdisc)/c2 ;  
        float root1p = root1 > t_min ? root1 : RT_DEFAULT_MAX ;   // disqualify -ve roots from mirror cone immediately 
        float root2p = root2 > t_min ? root2 : RT_DEFAULT_MAX ; 

        float t_near = fminf( root1p, root2p );
        float t_far  = fmaxf( root1p, root2p );  
        float z_near = o.z+t_near*d.z ; 
        float z_far  = o.z+t_far*d.z ; 

        t_near = z_near > z1 && z_near < z2  && t_near > t_min ? t_near : RT_DEFAULT_MAX ; // disqualify out-of-z
        t_far  = z_far  > z1 && z_far  < z2  && t_far  > t_min ? t_far  : RT_DEFAULT_MAX ; 

        float idz = 1.f/d.z ; 
        float t_cap1 = d.z == 0.f ? RT_DEFAULT_MAX : (z1 - o.z)*idz ;   // d.z zero means no z-plane intersects
        float t_cap2 = d.z == 0.f ? RT_DEFAULT_MAX : (z2 - o.z)*idz ;
        float r_cap1 = (o.x + t_cap1*d.x)*(o.x + t_cap1*d.x) + (o.y + t_cap1*d.y)*(o.y + t_cap1*d.y) ;  
        float r_cap2 = (o.x + t_cap2*d.x)*(o.x + t_cap2*d.x) + (o.y + t_cap2*d.y)*(o.y + t_cap2*d.y) ;  

        t_cap1 = r_cap1 < r1r1 && t_cap1 > t_min ? t_cap1 : RT_DEFAULT_MAX ;  // disqualify out-of-radius
        t_cap2 = r_cap2 < r2r2 && t_cap2 > t_min ? t_cap2 : RT_DEFAULT_MAX ; 
 
        float t_capn = fminf( t_cap1, t_cap2 );    // order caps
        float t_capf = fmaxf( t_cap1, t_cap2 );

        // NB use of RT_DEFAULT_MAX to represent disqualified
        // roots is crucial to picking closest  qualified root with 
        // the simple fminf(tt) 

        float4 tt = make_float4( t_near, t_far, t_capn, t_capf );
        float t_cand = fminf(tt) ; 
        
        valid_isect = t_cand > t_min && t_cand < RT_DEFAULT_MAX ;
        if(valid_isect)
        {
            if( t_cand == t_cap1 || t_cand == t_cap2 )
            {
                isect.x = 0.f ; 
                isect.y = 0.f ;
                isect.z =  t_cand == t_cap2 ? 1.f : -1.f  ;   
            }
            else
            { 
                //     x^2 + y^2  - (z - z0)^2 tanth^2 = 0 
                //     x^2 + y^2  - (z^2 -2z0 z - z0^2) tanth^2 = 0 
                //
                //   Gradient:    [2x, 2y, (-2z + 2z0) tanth^2 ] 
                //   Gradient:    2*[x, y, (z0-z) tanth^2 ] 
                float3 n = normalize(make_float3( o.x+t_cand*d.x, o.y+t_cand*d.y, (z0-(o.z+t_cand*d.z))*tth2  ))  ; 
                isect.x = n.x ; 
                isect.y = n.y ;
                isect.z = n.z ; 
            }
            isect.w = t_cand ; 
        }
    }
    return valid_isect ; 
}




INTERSECT_FUNC
bool intersect_node( float4& isect, const Prim* prim, const Node* node, const float4* plan, const float t_min , const float3& ray_origin, const float3& ray_direction )
{
    const unsigned typecode = node->typecode() ;  
    bool valid_isect = false ; 
    switch(typecode)
    {
        case CSG_SPHERE:           valid_isect = intersect_node_sphere(           isect, node->q0,               t_min, ray_origin, ray_direction ) ; break ; 
        case CSG_ZSPHERE:          valid_isect = intersect_node_zsphere(          isect, node->q0, node->q1,     t_min, ray_origin, ray_direction ) ; break ; 
        case CSG_CONVEXPOLYHEDRON: valid_isect = intersect_node_convexpolyhedron( isect, prim, node, plan,       t_min, ray_origin, ray_direction ) ; break ;
        case CSG_CONE:             valid_isect = intersect_node_cone(             isect, node->q0,               t_min, ray_origin, ray_direction ) ; break ;
    }
   return valid_isect ; 
}


