// ./intersect_tree.sh

#include <vector>
#include <iostream>
#include <iomanip>

#include "sutil_vec_math.h"
#include "OpticksCSG.h"
#include "Node.h"
#include "qat4.h"

#include "intersect_node.h"
#include "intersect_tree.h"

int main(int argc, char** argv)
{
    Node un = Node::Union() ; 
    Node bx = Node::Box3(90.f, 90.f, 90.f) ; 
    Node sp = Node::Sphere(100.f); 

    std::vector<Node> nds ; 
    nds.push_back(un) ;
    nds.push_back(bx) ; 
    nds.push_back(sp) ; 

    const Node* node = nds.data() ; 
    int numNode = nds.size() ; 

    const float4* plan0 = nullptr ; 
    const qat4* itra0 = nullptr ; 

    const float t_min = 0.f ; 
    const float3 ray_origin = make_float3( 0.f, 0.f, 0.f ); 
    const float3 ray_direction = make_float3( 0.f, 0.f, 1.f ); 
    float4 isect = make_float4(0.f, 0.f, 0.f, 0.f ); 

    bool valid_isect = intersect_tree( isect, numNode, node, plan0, itra0, t_min, ray_origin, ray_direction) ;

    std::cout 
        << "//intersect_tree.cc"
        << " valid_isect " << valid_isect
        << " isect "
        << " " << std::setw(10) << std::fixed << std::setprecision(4) << isect.x 
        << " " << std::setw(10) << std::fixed << std::setprecision(4) << isect.y 
        << " " << std::setw(10) << std::fixed << std::setprecision(4) << isect.z 
        << " " << std::setw(10) << std::fixed << std::setprecision(4) << isect.w 
        << std::endl
       ;    

    return 0 ; 
}

