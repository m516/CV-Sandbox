#pragma once

#include "../gameObject.h"

class TestCube : public GameObject{
    public:
    virtual void update();
    virtual size_t numVertices() const {return 36;}
};