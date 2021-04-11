
#pragma once

#include "../world.h"
#include "../gameObjects/test_cube.h"
#include "../gameObjects/test_triangle.h"
#include "../gameObjects/objFileObject.h"

#define NUM_BLOCKS 14
enum BlockType{GENERIC = 0, ROTATE_LEFT = 1, ROTATE_RIGHT = 2, GOAL = 3, ANTIGRAVITY = 4};
int blocks[NUM_BLOCKS][4] = {
//  x   y   z, type
   {0 , -1,  0, 0 },
   {2 , -2, -3, 3 },
   {-1, -1,  0, 0 },
   {1 , -2, -3, 0 },
   {-4, -2,  1, 0 },
   {-2, -1,  0, 0 },
   {0 , -2, -3, 0 },
   {-4, -2,  0, 0 },
   {-1, -2, -3, 0 },
   {-4, -2, -1, 2 },
   {-2, -2, -3, 0 },
   {-4, -2, -2, 0 },
   {-3, -2, -3, 0 },
   {-4, -2, -3, 0 }
};

/**
 * @brief The first level of the WYSIWYG game
 */
class World4_WYSIWYG : public World{
    public:
    World4_WYSIWYG(VertexArray &vertexArray) : World(vertexArray){
        //Load the basic passthrough shader
        //Keep a reference to the actual MVP shader to expose camera-related parameters.
        _mvpShader = new Shaders::TexturedMVP();
        _shader = _mvpShader;
        //Populate the world with blocks
        ObjFileObject* genericBlockTemplate = new ObjFileObject("block.obj");
        ObjFileObject* specialBlockTemplate = new ObjFileObject("block2.obj");
        for(int i = 0; i < NUM_BLOCKS; i++){
            if(blocks[i][3]==BlockType::GENERIC){
                genericBlockTemplate->_x = blocks[i][0];
                genericBlockTemplate->_y = blocks[i][1];
                genericBlockTemplate->_z = blocks[i][2];
                _registerGameObject(genericBlockTemplate);
                genericBlockTemplate = genericBlockTemplate->copy();
            }
            else{ //This is special
                specialBlockTemplate->_x = blocks[i][0];
                specialBlockTemplate->_y = blocks[i][1];
                specialBlockTemplate->_z = blocks[i][2];
                //Color the block based on its type
                switch(blocks[i][3]){
                    case BlockType::ROTATE_LEFT:
                        specialBlockTemplate->_r = 1;
                        specialBlockTemplate->_g = 0;
                        specialBlockTemplate->_b = 0;
                        break;
                    case BlockType::ROTATE_RIGHT:
                        specialBlockTemplate->_r = 0;
                        specialBlockTemplate->_g = 0;
                        specialBlockTemplate->_b = 1;
                        break;
                    case BlockType::GOAL:
                        specialBlockTemplate->_r = 0;
                        specialBlockTemplate->_g = 1;
                        specialBlockTemplate->_b = 0;
                        break;
                    case BlockType::ANTIGRAVITY:
                        specialBlockTemplate->_r = 0;
                        specialBlockTemplate->_g = 1;
                        specialBlockTemplate->_b = 1;
                        break;
                    default:
                        specialBlockTemplate->_r = 1;
                        specialBlockTemplate->_g = 0;
                        specialBlockTemplate->_b = 1;
                        break;
                }
                _registerGameObject(specialBlockTemplate);
                specialBlockTemplate = specialBlockTemplate->copy();
            }
        }
        delete genericBlockTemplate;
        //Create a block for the player.
        _player = specialBlockTemplate;
        _player->_r = 1;
        _player->_g = .5;
        _player->_b = .1;
        _registerGameObject(_player);
    }

    virtual void update(){
        static bool keyInput[4];     //previous keyboard right, up, left, down
        static bool currKeyInput[4]; //Keyboard right, up, left, down
        //Player behavior
        if(_renderer != NULL && _player != NULL) {
            //Turn on sticky keys so players don't go zoom
            glfwSetInputMode(_renderer->window(), GLFW_STICKY_KEYS, GL_TRUE);
            currKeyInput[0] = glfwGetKey(_renderer->window(), GLFW_KEY_A) == GLFW_PRESS;
            currKeyInput[1] = glfwGetKey(_renderer->window(), GLFW_KEY_W) == GLFW_PRESS;
            currKeyInput[2] = glfwGetKey(_renderer->window(), GLFW_KEY_D) == GLFW_PRESS;
            currKeyInput[3] = glfwGetKey(_renderer->window(), GLFW_KEY_S) == GLFW_PRESS;
            if(currKeyInput[0] && !keyInput[0]) _player->_x++;
            if(currKeyInput[1] && !keyInput[1]) _player->_z++;
            if(currKeyInput[2] && !keyInput[2]) _player->_x--;
            if(currKeyInput[3] && !keyInput[3]) _player->_z--;
            for(int i = 0; i < 4; i++) keyInput[i] = currKeyInput[i];
            _mvpShader->cameraPosition[0] = (_mvpShader->cameraPosition[0]+(_player->_x-5*_mvpShader->cameraDirection[0])*_deltaTime)/(1+_deltaTime);
            _mvpShader->cameraPosition[1] = (_mvpShader->cameraPosition[1]+(_player->_y-5*_mvpShader->cameraDirection[1])*_deltaTime)/(1+_deltaTime);
            _mvpShader->cameraPosition[2] = (_mvpShader->cameraPosition[2]+(_player->_z-5*_mvpShader->cameraDirection[2])*_deltaTime)/(1+_deltaTime);
        }
        World::update();
    }

    private:
        Shaders::TexturedMVP *_mvpShader;
        ObjFileObject *_player;
};