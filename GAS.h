#pragma once

#include <vector>
#include "AS.h"
#include "BI.h"

struct Solid ; 

struct GAS : public AS
{
    const Solid*    so ; 
    std::vector<BI> bis ; 
};



