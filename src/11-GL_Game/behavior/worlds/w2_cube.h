
#pragma once

#include "../world.h"
#include "../gameObjects/test_cube.h"
#include "../gameObjects/test_triangle.h"

class World2_CubeDemo : public World{
    public:
    World2_CubeDemo(VertexArray &vertexArray) : World(vertexArray){
        //Load the basic passthrough shader
        _shader = new Shaders::MVP();
        //Add a single cube to the world
        TestCube* c = new TestCube();
        _registerGameObject(c);
        //Add a single triangle to the world
        TestTriangle* t = new TestTriangle();
        _registerGameObject(t);
    }

    virtual void update(){
        World::update();
    }
};