
#pragma once

#include "../world.h"
#include "../gameObjects/test_cube.h"
#include "../gameObjects/test_triangle.h"
#include "../gameObjects/objFileObject.h"

/**
 * @brief Demonstrates multiple objects loaded dynamically from a file
 */
class World3_Objects : public World{
    public:
    World3_Objects(VertexArray &vertexArray) : World(vertexArray){
        //Load the basic passthrough shader
        _shader = new Shaders::TexturedMVP();
        //Add a single cube to the world
        ObjFileObject* c = new ObjFileObject("block.obj");
        _registerGameObject(c);
        ObjFileObject* c2 = c->copy();
        c2->_x = 1;
        _registerGameObject(c2);
        c = new ObjFileObject("block2.obj");
        c->_x = 1;
        c->_z = 2;
        _registerGameObject(c);
        //Add a single triangle to the world
        TestTriangle* t = new TestTriangle();
        _registerGameObject(t);
    }

    virtual void update(){
        World::update();
    }
};