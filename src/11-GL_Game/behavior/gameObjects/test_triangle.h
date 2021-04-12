#pragma once

#include "../gameObject.h"

class TestTriangle : public GameObject{
    public:
    virtual void update();
    virtual size_t numVertices() const {return 3;}
};