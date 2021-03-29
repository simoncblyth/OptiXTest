#if defined(__CUDACC__) || defined(__CUDABE__)
#else

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector_types.h>

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
#endif

