#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector_types.h>

#include "sutil_vec_math.h"


#include "OpticksCSG.h"
#include "Node.h"

std::string Node::desc() const 
{
    std::stringstream ss ; 
    ss
       << "Node "
       << CSG::Name((OpticksCSG_t)typecode())
       ;    
    std::string s = ss.str();
    return s ; 
}


void Node::Dump(const Node* n_, unsigned ni, const char* label)
{
    std::cout << "Node::Dump ni " << ni << " " ; 
    if(label) std::cout << label ;  
    std::cout << std::endl ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
        const Node* n = n_ + i ;        
        std::cout << "(" << i << ")" << std::endl ; 
        std::cout 
            << " node.q0.f.xyzw ( " 
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q0.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q0.f.y  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q0.f.z  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q0.f.w
            << " ) " 
            << std::endl 
            << " node.q1.f.xyzw ( " 
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q1.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q1.f.y  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q1.f.z  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q1.f.w
            << " ) " 
            << std::endl 
            << " node.q2.f.xyzw ( " 
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q2.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q2.f.y  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q2.f.z  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q2.f.w
            << " ) " 
            << std::endl 
            << " node.q3.f.xyzw ( " 
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q3.f.x  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q3.f.y  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q3.f.z  
            << std::setw(10) << std::fixed << std::setprecision(3) << n->q3.f.w
            << " ) " 
            << std::endl 
            ;

        std::cout 
            << " node.q0.i.xyzw ( " 
            << std::setw(10) << n->q0.i.x  
            << std::setw(10) << n->q0.i.y  
            << std::setw(10) << n->q0.i.z  
            << std::setw(10) << n->q0.i.w
            << " ) " 
            << std::endl 
            << " node.q1.i.xyzw ( " 
            << std::setw(10) << n->q1.i.x  
            << std::setw(10) << n->q1.i.y  
            << std::setw(10) << n->q1.i.z  
            << std::setw(10) << n->q1.i.w
            << " ) " 
            << std::endl 
            << " node.q2.i.xyzw ( " 
            << std::setw(10) << n->q2.i.x  
            << std::setw(10) << n->q2.i.y  
            << std::setw(10) << n->q2.i.z  
            << std::setw(10) << n->q2.i.w
            << " ) " 
            << std::endl 
            << " node.q3.i.xyzw ( " 
            << std::setw(10) << n->q3.i.x  
            << std::setw(10) << n->q3.i.y  
            << std::setw(10) << n->q3.i.z  
            << std::setw(10) << n->q3.i.w
            << " ) " 
            << std::endl 
            ;
    }
}


Node Node::Sphere(float radius)  // static
{
    assert( radius > 0.f); 
    Node nd = {} ;
    nd.setParam( 0.f, 0.f, 0.f, radius,  0.f,  0.f ); 
    nd.setAABB(  -radius, -radius, -radius,  radius, radius, radius  ); 
    nd.setTypecode(CSG_SPHERE) ; 
    return nd ;
}

Node Node::ZSphere(float radius, float z1, float z2)  // static
{
    assert( radius > 0.f); 
    assert( z2 > z1 ); 
    Node nd = {} ;
    nd.setParam( 0.f, 0.f, 0.f, radius, z1, z2 ); 
    nd.setAABB(  -radius, -radius, z1,  radius, radius, z2  ); 
    nd.setTypecode(CSG_ZSPHERE) ; 
    return nd ;
}

Node Node::Cone(float r1, float z1, float r2, float z2)  // static
{
    assert( z2 > z1 ); 
    float rmax = fmaxf(r1, r2) ;
    Node nd = {} ;
    nd.setParam( r1, z1, r2, z2, 0.f, 0.f ) ;
    nd.setAABB( -rmax, -rmax, z1, rmax, rmax, z2 ); 
    nd.setTypecode(CSG_CONE) ; 
    return nd ; 
}

Node Node::Hyperboloid(float r0, float zf, float z1, float z2) // static
{
    assert( z1 < z2 ); 
    const float rr0 = r0*r0 ; 
    const float z1s = z1/zf ; 
    const float z2s = z2/zf ; 

    const float rr1 = rr0 * ( z1s*z1s + 1.f ) ;
    const float rr2 = rr0 * ( z2s*z2s + 1.f ) ;
    const float rmx = sqrtf(fmaxf( rr1, rr2 )) ; 


    Node nd = {} ;
    nd.setParam(r0, zf, z1, z2, 0.f, 0.f ) ; 
    nd.setAABB(  -rmx,  -rmx,  z1,  rmx, rmx, z2 ); 
    nd.setTypecode(CSG_HYPERBOLOID) ; 
    return nd ; 
}

Node Node::Box3(float fx, float fy, float fz )  // static 
{
    assert( fx > 0.f ); 
    assert( fy > 0.f ); 
    assert( fz > 0.f ); 

    Node nd = {} ;
    nd.setParam( fx, fy, fz, 0.f, 0.f, 0.f ); 
    nd.setAABB( -fx*0.5f , -fy*0.5f, -fz*0.5f, fx*0.5f , fy*0.5f, fz*0.5f );   
    nd.setTypecode(CSG_BOX3) ; 
    return nd ; 
}

Node Node::Plane(float nx, float ny, float nz, float d)
{
    Node nd = {} ;
    nd.setParam(nx, ny, nz, d, 0.f, 0.f ) ;
    nd.setTypecode(CSG_PLANE) ; 
    return nd ; 
}

Node Node::Slab(float nx, float ny, float nz, float d1, float d2 )
{
    Node nd = {} ;
    nd.setParam( nx, ny, nz, 0.f, d1, d2 ); 
    nd.setTypecode(CSG_SLAB) ; 
    return nd ; 
}


Node Node::Cylinder(float px, float py, float radius, float z1, float z2)
{
    Node nd = {} ; 
    nd.setParam( px, py, 0.f, radius, z1, z2)  ; 
    nd.setAABB( px-radius, py-radius, z1, px+radius, py+radius, z2 );   
    nd.setTypecode(CSG_CYLINDER); 
    return nd ; 
} 

Node Node::Disc(float px, float py, float ir, float r, float z1, float z2)
{
    Node nd = {} ;
    nd.setParam( px, py, ir, r, z1, z2 ); 
    nd.setAABB( px - r , py - r , z1, px + r, py + r, z2 ); 
    nd.setTypecode(CSG_DISC); 
    return nd ; 
} 


 



#endif

