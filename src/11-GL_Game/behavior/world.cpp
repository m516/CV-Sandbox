#include <chrono>

#include "world.h"
#include "gameObject.h"
#include "../engine/shader.h"


World::World(VertexArray &vertexArray){
    _vertexArray = &vertexArray;
}

void World::_registerGameObject(GameObject * o){
    if(!_vertexArray->resizable()) RUNTIME_ERROR("Cannot resize vertex array to accomodate for new objects.\nMake sure all objects are added before the first draw call.");
    if(_vertexArray == NULL)       RUNTIME_ERROR("Null vertex array");
    _objects.push_back(o);
    size_t start = _vertexArray->size();
    _vertexArray->add(o->numVertices());
    o->_vertexArray = _vertexArray->subset(start);
    o->_world       = this;
}

void World::update(){
    _updateTime();
    for(int i = 0; i < _objects.size(); i++){
        _objects[i]->update();
    }
}

void World::_updateTime(){
    typedef std::chrono::microseconds microseconds;

    static auto previousTime = std::chrono::steady_clock::now();
    static auto gameStartTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    
    microseconds dt = std::chrono::duration_cast<microseconds>(currentTime - previousTime);
    microseconds gt = std::chrono::duration_cast<microseconds>(currentTime - gameStartTime);

    _deltaTime = (float)dt.count();
    _gameTime  = (float)gt.count();

    _deltaTime /= 1e6;
    _gameTime /= 1e6;

    previousTime = currentTime;
}

void World::_loadDefaultShader(){
    _shader = new Shaders::Passthrough();
}

