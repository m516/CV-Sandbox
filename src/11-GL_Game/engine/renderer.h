#pragma once

#include "../behavior/world.h"
#include "../utils.h"

class World;
class Shader;

class Renderer{
    public:
    VertexArray vertexArray;
    /**
     * @brief Creates a new GLFW window that the world can manipulate.
     * The world is used to keep track of the shader.
     */
    void init(World *world);
    /**
     * @brief Renders the world on the screen
     */
    void render();
    void close();

    private:
    GLFWwindow* _window = NULL;
    Shaders::AbstractShader *_shader;
    World *_world;
};

