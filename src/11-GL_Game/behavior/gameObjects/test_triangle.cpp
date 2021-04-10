#include "test_triangle.h"

// Just a single triangle for GL_TRIANGLES to test. Add to this to draw more points
const float vertex_data[] = { 
        -0.5f, -0.5f, 0.0f,
        0.0f,   0.5f, 0.0f,
        0.5f,  -0.5f, 0.0f,
};

// One color (RGB for this) for each vertex.
const float color_data[] = { 
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f,
};

void TestTriangle::update(){
    _y = cos(_world->gameTime());
    _x = sin(_world->gameTime());

    for(int i = 0; i < numVertices(); i++){
        for(int j = 0; j < 3; j++){
            _vertexArray[i]->position[j]=vertex_data[i*3+j]+_pos[j];
            _vertexArray[i]->color[j]   =color_data[i*3+j];
        }
    }
}
