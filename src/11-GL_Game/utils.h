#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define RUNTIME_ERROR(...){                                           \
fprintf(stderr, "Uh oh, runtime error!\n");                           \
fprintf(stderr, "Encountered on %s:%d\n", __FILE__, __LINE__);        \
fprintf(stderr, "In function %s\n", __func__);                        \
fprintf(stderr, __VA_ARGS__);                                         \
fprintf(stderr, "\n");                                                \
terminate(1);                                                         \
} 

using namespace std;


/**
 * A helper function for terminating the program
 */
void terminate(int errorCode);

/**
 * A callback function for GLFW to execute when an internal error occurs with the
 * library.
 */
void error_callback(int error, const char* description);



/**
 * @brief A value that synchronizes with a GLFloat
 */
class GLValue{
    public:
        bool operator==   (float i){ return i   ==_t; }
        bool operator== (GLValue i){ return i._t==_t; }
        bool operator!=   (float i){ return i   !=_t; }
        bool operator!= (GLValue i){ return i._t!=_t; }
        void operator=    (float i){ if(i != _t){ _t = i;    _dirty = true; } }
        void operator=  (GLValue i){ if(i != _t){ _t = i._t; _dirty = true; } if(_v == nullptr) _v = i._v;  if(i._dirty) _dirty = true; }
        operator float()           { return _t; }
        bool dirty(){ return _dirty; }
    private:
        friend class Vertex;
        friend class VertexArray;
        float _t = 0;
        GLfloat* _v = nullptr;
        bool _dirty = true;
        void _sync(){
            if(_v==nullptr) RUNTIME_ERROR("Attempting to sync a GLValue that hasn't been paired with a GL float");
            if(!_dirty) return;
            *_v = _t;
            _dirty = false;
        }
};

class Vertex{
    public:
    GLValue color[3], position[3];
    GLValue &x = position[0], &y = position[1], &z = position[2];
    GLValue &r = color[0], &g = color[1], &b = color[2];
};

class VertexArray{
    public:
    size_t size(){return _vertices.size();}
    void add(size_t num_new_vertices){
        //Can't add any more vertices when the buffers have been initialized.
        if(_boundToGL()) RUNTIME_ERROR("Attempting to modify a VertexArray that has been synchronized with OpenGL data");
        for(int i = 0; i < num_new_vertices; i++){
            _vertices.push_back(new Vertex());
        }
    }
    VertexArray subset(size_t start, size_t end = -1){
        //Sanity checking
        if(start<0)       RUNTIME_ERROR("Start is %zu, less than 0", start);
        if(start>=size()) RUNTIME_ERROR("Start is %zu, greater than or equal to size %zu", start, size());
        if(end>0xFFFFFFFF)         end = size();
        if(end>size())    RUNTIME_ERROR("end   is %zu, greater than size %zu", end, size());
        VertexArray v;
        //Vertex array
        auto first = _vertices.begin() + start;
        auto last = _vertices.begin() + end;
        v._vertices = vector<Vertex*>(first, last);
        //Metadata
        v._vertexColorBuffer        = _vertexColorBuffer;
        v._vertexPositionBuffer     = _vertexPositionBuffer;
        v._vertexColorBufferData    = _vertexColorBufferData    + 3*start;
        v._vertexPositionBufferData = _vertexPositionBufferData + 3*start;
        return v;
    }
    ~VertexArray(){
        if(_allocatedData){
            delete _vertexColorBufferData;
            delete _vertexPositionBufferData;
        }
    }
    bool resizable(){ return !_boundToGL(); }
    Vertex *operator[](size_t index){
        if(index>size()) RUNTIME_ERROR("Invalid index: given %zu but the length of the array is %zu", index, size());
        return _vertices[index];
    }
    private:
    friend class Renderer;
    bool _allocatedData = false;
    vector<Vertex*> _vertices;
    GLuint _vertexColorBuffer, _vertexPositionBuffer, _vertexArray;
    GLfloat *_vertexColorBufferData = nullptr, *_vertexPositionBufferData = nullptr;
    bool _boundToGL(){ return _vertexPositionBufferData != nullptr; }
    void _allocateBuffers(){
        //Check for potential memory leaks caused by allocating buffer more than once
        if(_boundToGL()) RUNTIME_ERROR("Attempting to modify a VertexArray that has been synchronized with OpenGL data");
        //Vertex array
        glGenVertexArrays(1, &_vertexArray);
        glBindVertexArray(_vertexArray);
        //Allocate memory for vertex positions and colors
        glGenBuffers(1, &_vertexPositionBuffer);
        glGenBuffers(1, &_vertexColorBuffer);
        //Create GLfloat arrays
        _vertexColorBufferData = new GLfloat[size()*3];
        _vertexPositionBufferData = new GLfloat[size()*3];
        //Set allocated flag to ensure data gets destroyed when the vertex array is destroyed
        _allocatedData = true;
        //Bind all vertices
        GLfloat *caddr = _vertexColorBufferData, *paddr = _vertexPositionBufferData;
        for(int i = 0; i < size(); i++){
            _vertices[i]->color[0]._v    = caddr++;
            _vertices[i]->position[0]._v = paddr++;
            _vertices[i]->color[1]._v    = caddr++;
            _vertices[i]->position[1]._v = paddr++;
            _vertices[i]->color[2]._v    = caddr++;
            _vertices[i]->position[2]._v = paddr++;
        }
    }
    void _sync(){
        //Can't sync when not bound to OpenGL float array
        if(!_boundToGL()) RUNTIME_ERROR("Attempting to sync a VertexArray that has not been synchronized with OpenGL data");
        //TODO only synchronize dirty vertices
        for(int i = 0; i < size(); i++){
            _vertices[i]->color[0]._sync();
            _vertices[i]->color[1]._sync();
            _vertices[i]->color[2]._sync();
            _vertices[i]->position[0]._sync();
            _vertices[i]->position[1]._sync();
            _vertices[i]->position[2]._sync();
        }
    }
};
