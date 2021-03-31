// ./CUTest.sh 

#include <vector>
#include <iostream>

#include "sutil_vec_math.h"
#include "Quad.h"
#include "qat4.h"
#include "CU.h"

int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ;     

    std::vector<qat4> qq ;  
    for(unsigned i=0 ; i < 10 ; i++)
    {
        qat4 q ; 
        q.q0.i.x = i ; 
        q.q1.i.x = i*10 ;  
        q.q2.i.x = i*100 ;  
        q.q3.i.x = i*1000 ;  
        qq.push_back(q); 
    }

    unsigned num_q = qq.size() ; 

    qat4* d_qq = CU::UploadArray<qat4>(qq.data(), num_q ); 

    qat4* qq2 = CU::DownloadArray<qat4>(d_qq, num_q );     

    for(unsigned i=0 ; i < num_q ; i++)
    {
        const qat4& q = qq2[i] ; 
        std::cout << i << std::endl ; 
        std::cout << q << std::endl ; 
    } 

    return 0 ; 
}


