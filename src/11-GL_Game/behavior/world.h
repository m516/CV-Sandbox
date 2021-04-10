#pragma once

#include <vector>

#include "../utils.h"
#include "../engine/shader.h"

class AbstractShader;

class GameObject;

class World{
    public:
    World(VertexArray &vertexArray);
    /**
     * @brief Difference of time between the current draw call and the previous one in seconds.
     * 
     * @return float 
     */
    float deltaTime(){return _deltaTime;}
    /**
     * @brief Differene of time between current draw call and the first one, in seconds.
     * 
     * @return float 
     */
    float gameTime(){return _gameTime;}
    /**
     * @brief Updates all vertices
     * 
     */
    virtual void update();
    virtual ~World(){
        delete _shader;
        for(int i = 0; i < _objects.size(); i++){
            delete _objects[i];
        }
    }
    protected:
    friend class Renderer;
    std::vector<GameObject*> _objects;
    VertexArray* _vertexArray = NULL;
    /**
     * @brief Registers a game object with this world. 
     * All registered GameObjects will be deleted as soon
     * as the destructor for this World instance is called,
     * so make sure to create new GameObject instances.
     *  
     * @param o the GameObject to add to the world.
     */
    void _registerGameObject(GameObject * o);
    float _deltaTime;
    float _gameTime;
    void _updateTime();
    Shaders::AbstractShader* _shader;
    /**
     * @brief Loads the passthrough shader. A template for loading shaders in worlds.
     */
    void _loadDefaultShader();
};