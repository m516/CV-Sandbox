#pragma once

#include "../world.h"
#include "../gameObjects/test_triangle.h"

class World1_TriangleDemo : public World{
    public:
    World1_TriangleDemo(VertexArray &vertexArray) : World(vertexArray){
        //Load the basic passthrough shader
        _shader = new Shaders::Passthrough();
        //Add a single triangle to the world
        TestTriangle* t = new TestTriangle();
        _registerGameObject(t);
    }
};