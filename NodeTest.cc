// name=NodeTest ; gcc $name.cc Node.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name 

#include <vector>
#include <iomanip>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector_types.h>
#include "Node.h"
#include "Sys.h"

int main(int argc, char** argv)
{
   std::cout << argv[0] << std::endl ; 

   glm::mat4 m0(1.f); 
   glm::mat4 m1(2.f); 
   glm::mat4 m2(3.f); 

   m0[0][3] = Sys::int_as_float(42); 
   m1[0][3] = Sys::int_as_float(52); 
   m2[0][3] = Sys::int_as_float(62); 

   std::vector<glm::mat4> node ; 
   node.push_back(m0); 
   node.push_back(m1); 
   node.push_back(m2); 

   std::vector<Node> node_(3) ; 

   memcpy( node_.data(), node.data(), sizeof(Node)*node_.size() );  

   Node* n_ = node_.data(); 
   Node::Dump( n_, node_.size(), "NodeTest" );  

   return 0 ; 
}
