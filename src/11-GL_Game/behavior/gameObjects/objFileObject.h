#pragma once

#include "../gameObject.h"

class ObjFileObject : public GameObject{
    public:
    ObjFileObject(){}
    ObjFileObject(const char* filename);
    virtual void update();
    virtual size_t numVertices() const {return _numVertices;}
    virtual ~ObjFileObject(){
        if(_ownsVertices){
            delete _vertexPositions;
            delete _vertexUVs;
            delete _vertexColors;
            delete _vertexNormals;
        }
    }
    void operator=(ObjFileObject other){
        if(!_vertexArray.resizable()) RUNTIME_ERROR("Attempting to modify an object that isn't resizable");

        _vertexColors    = other._vertexColors;
        _vertexNormals   = other._vertexNormals;
        _vertexPositions = other._vertexPositions;
        _vertexUVs       = other._vertexUVs;
        _vertexColors    = other._vertexColors;
        _numVertices     = other._numVertices;
        _ownsVertices    = false;
    }
    ObjFileObject* copy(){
        ObjFileObject* o = new ObjFileObject;
        o->_vertexPositions  = _vertexPositions;
        o->_vertexColors     = _vertexColors;
        o->_vertexNormals    = _vertexNormals;
        o->_vertexUVs        = _vertexUVs;
        o->_vertexColors     = _vertexColors;
        o->_numVertices      = _numVertices;
        o->_ownsVertices     = false;
        return o;
    }
    private:
    float* _vertexPositions;
    float* _vertexColors;
    float* _vertexUVs;
    float* _vertexNormals;
    size_t _numVertices = 0;
    bool _ownsVertices = false;
    
};