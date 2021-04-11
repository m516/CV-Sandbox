#pragma once

#include "../utils.h"
#include "world.h"

class GameObject{
    public:
    /**
     * @brief Called once per frame; allows the GameObjet to update the vertices how it sees fit
     * 
     */
    virtual void update() = 0;
    /**
     * @brief Returns the number of vertices this gameObject requires. 
     * This value should not change because no more vertices can be given to this GameObject during the draw call.
     * 
     * @return size_t 
     */
    virtual size_t numVertices() const = 0;
    virtual ~GameObject(){}
    float _pos[3] = {0, 0, 0};
    float &_x = _pos[0], &_y = _pos[1], &_z = _pos[2];
    protected:
    friend class World;
    VertexArray _vertexArray;
    World* _world;
};