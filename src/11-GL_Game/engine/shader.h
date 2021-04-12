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


#include "stb_image.h"
#include "../utils.h"


class Renderer;

namespace Shaders{

    GLuint Load(const char * vertex_file_path,const char * fragment_file_path);

    class AbstractShader{
        public:
        virtual ~AbstractShader(){}
        protected:
        friend class Renderer;
        Renderer *_renderer;
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

    class TexturedMVP : public AbstractShader{
        public:
        float cameraPosition[3] = {-2, 2, -2};
        float cameraDirection[3] = {-1, -1, -1};
        float zoom = 5;
        /**
         * @brief Returns the most recent view matrix
         * @return glm::mat4 
         */
        glm::mat4 M() {return _M;}
        ~TexturedMVP(){ }
        private:
        GLuint _viewMatrixID;
        int _textureWidth, _textureHeight, _textureChannels;
        GLuint _textureID;
        glm::mat4 _M;
        

        virtual void _init(){
            //Create a new OpenGL texture
            glGenTextures(1, &_textureID); 
            glBindTexture(GL_TEXTURE_2D, _textureID); 
            // set the texture wrapping/filtering options (on the currently bound texture object)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);	
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            //Load the texture
            unsigned char* textureData;
            const char* texturePath = MEDIA_DIRECTORY "texturedBlocks.png";
            textureData = stbi_load(texturePath, &_textureWidth, &_textureHeight, &_textureChannels, 0);
            printf("Texture: %d x %d x %d\n", _textureWidth, _textureHeight, _textureChannels);
            if(!textureData) RUNTIME_ERROR("Failed to load texture at '%s'\n", texturePath); 
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _textureWidth, _textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
            stbi_image_free(textureData);
            //Load the shaders
            _vertexShaderFilename = "shaders/textured_mvp_transform.vert";
            _fragmentShaderFilename = "shaders/texture.frag";
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
            //Calculate the projection matrix.
            //Assumes a square (1:1) aspect ratio
            glm::mat4 projectionMatrix = glm::ortho(-zoom, zoom, -zoom, zoom, -100.f, 100.f); 
            // glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
            _M = projectionMatrix * viewMatrix;
            //Apply the matrix
            glUniformMatrix4fv(_viewMatrixID, 1, GL_FALSE, &_M[0][0]);
        }
    };
}

