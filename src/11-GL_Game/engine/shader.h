#pragma once

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

#include <stdlib.h>
#include <string.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Shaders{

    GLuint Load(const char * vertex_file_path,const char * fragment_file_path);

    class AbstractShader{
        protected:
        friend class Renderer;
        GLuint _programID;
        std::string _vertexShaderFilename;
        std::string _fragmentShaderFilename;
        std::string _genShaderPath(const std::string &shaderFileName);
        /**
         * @brief Create the shader. Run once before the first draw call.
         */
        virtual void _init();
        /**
         * @brief Loads the shaders from the path
         * 
         */
        void _loadShaders();
        /**
         * @brief Apply the shader. Run at the beginning of every draw call.
         * 
         */
        virtual void _apply();
    };

    class Passthrough : public AbstractShader{
        private:
        virtual void _init(){
            _vertexShaderFilename = "shaders/passthrough.vert";
            _fragmentShaderFilename = "shaders/passthrough.frag";
            _loadShaders();
        }
    };

    class MVP : public AbstractShader{
        public:
            float cameraPosition[3] = {-2, -2, -2};
            float cameraDirection[3] = {1, 1, 1};
        private:
        GLuint _viewMatrixID;
        virtual void _init(){
            _vertexShaderFilename = "shaders/mvp_transform.vert";
            _fragmentShaderFilename = "shaders/passthrough.frag";
            _loadShaders();
            // Send our uniforms to the currently bound shader
	        _viewMatrixID = glGetUniformLocation(_programID, "MVP");
            // Enable depth test
            glEnable(GL_DEPTH_TEST);
            // Accept fragment if it closer to the camera than the former one
            glDepthFunc(GL_LESS);
        }
        virtual void _apply(){
            //Call super class apply method
            AbstractShader::_apply();
            //Recalculate the view matrix
            glm::vec3 _eye(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
            glm::vec3 _center(cameraPosition[0]+cameraDirection[0], 
                              cameraPosition[1]+cameraDirection[1], 
                              cameraPosition[2]+cameraDirection[2]);
            glm::vec3 _up(0, 1, 0);
            glm::mat4 viewMatrix = glm::lookAt(_eye, _center, _up);
            glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
            glm::mat4 M = projectionMatrix * viewMatrix;
            //Apply the matrix
            glUniformMatrix4fv(_viewMatrixID, 1, GL_FALSE, &M[0][0]);
        }
    };
}

