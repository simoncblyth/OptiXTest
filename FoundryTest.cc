// ./FoundryTest.sh

#include <iostream>
#include <cassert>

#include "sutil_vec_math.h"
#include "Foundry.h"


void test_layered()
{
    Foundry fd ;  

    Solid* s0 = fd.makeLayered("sphere", 100.f, 10 ); 
    Solid* s1 = fd.makeLayered("sphere", 1000.f, 10 ); 
    Solid* s2 = fd.makeLayered("sphere", 50.f, 5 ); 
    Solid* s3 = fd.makeSphere() ; 

    fd.dump(); 

    assert( fd.getSolidIdx(s0) == 0 ); 
    assert( fd.getSolidIdx(s1) == 1 ); 
    assert( fd.getSolidIdx(s2) == 2 ); 
    assert( fd.getSolidIdx(s3) == 3 ); 

    fd.write("/tmp", "FoundryTest_" ); 
}

void test_PrimSpec()
{
    Foundry fd ; 
    fd.makeDemoSolids(); 
    for(unsigned i = 0 ; i < fd.solid.size() ; i++ )
    {
        unsigned solidIdx = i ; 
        std::cout << "solidIdx " << solidIdx << std::endl ; 
        PrimSpec ps = fd.getPrimSpec(solidIdx);
        ps.dump(""); 
    }

    std::string bmap = fd.getBashMap(); 
    std::cout  << bmap << std::endl ; 
}

void test_addTran()
{
    Foundry fd ; 
    const Tran<double>* tr = Tran<double>::make_translate( 100., 200., 300. ) ; 
    unsigned idx = fd.addTran( *tr ); 
    assert( idx == 1u ); 
    const qat4* t = idx == 0 ? nullptr : fd.getTran(idx-1u) ; 
    const qat4* v = idx == 0 ? nullptr : fd.getItra(idx-1u) ; 
 
    std::cout << "idx " << idx << std::endl ; 
    std::cout << "t" << *t << std::endl ; 
    std::cout << "v" << *v << std::endl ; 
}

void test_makeClustered()
{
    std::cout << "[test_makeClustered" << std::endl ; 
    Foundry fd ; 
    fd.makeClustered("sphe", -1,2,1, -1,2,1, -1,2,1, 1000. ); 
    fd.dumpPrim(0); 
    std::cout << "]test_makeClustered" << std::endl ; 
}





int main(int argc, char** argv)
{
    //test_layered(); 
    //test_PrimSpec(); 
    //test_addTran(); 
    test_makeClustered(); 

    return 0 ; 
}
