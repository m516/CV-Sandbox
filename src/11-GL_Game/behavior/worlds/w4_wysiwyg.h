
#pragma once

#include "../world.h"
#include "../gameObjects/test_cube.h"
#include "../gameObjects/test_triangle.h"
#include "../gameObjects/objFileObject.h"

#define PLAYER_TRANSLATION_SPEED 50
#define CAMERA_TRANSLATION_SPEED 1.5
#define CAMERA_ROTATION_SPEED 2

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
                genericBlockTemplate->_x = (float)blocks[i][0];
                genericBlockTemplate->_y = (float)blocks[i][1];
                genericBlockTemplate->_z = (float)blocks[i][2];
                _registerGameObject(genericBlockTemplate);
                genericBlockTemplate = genericBlockTemplate->copy();
            }
            else{ //This is special
                specialBlockTemplate->_x = (float)blocks[i][0];
                specialBlockTemplate->_y = (float)blocks[i][1];
                specialBlockTemplate->_z = (float)blocks[i][2];
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

    void move(int* playerPos, int direction, int playerDirection, bool reverse = false){
        //A LUT that can be used to find the player's motion by summing its direction with the "direction" variable.
        //The player only moves along the x and z plane, so the y-value is only to support antigravity in the future.
        const int motion[4][3] = {
            { 0, -1, -1}, //W
            {-1, -1,  0}, //Q
            { 0, -1,  1}, //A
            { 1, -1,  0}  //S
        };
        int relativeMotion = (playerDirection + direction + (reverse?2:0))%4;
        if(relativeMotion<0) relativeMotion += 4;
        printf("Moving player. \n");
        printf("\tCamera:          %d\n", direction);
        printf("\tPlayer direction: %d\n", playerDirection);
        printf("\tRelative motion:  %d\n", relativeMotion);
        printf("\tPlayer position:  (%d,%d,%d)\n", playerPos[0], playerPos[1], playerPos[1]);
        playerPos[0] += motion[relativeMotion][0];
        playerPos[2] += motion[relativeMotion][2];
        printf("\tUpdated Player position:  (%d,%d,%d)\n", playerPos[0], playerPos[1], playerPos[1]);
    }

    //Snaps the player so it's on top of a block. Returns the index of the block if one could be found, otherwise it returns -1.
    int snap(int* playerPos, bool antigravity = false, float threshold = 0.01){
        if(antigravity) playerPos[2]++;
        else            playerPos[1]--;
        glm::vec4 p(playerPos[0], playerPos[1], playerPos[2], 1);
        p = _mvpShader->M()*p;
        //Use the shader's matrix to find other blocks in the same location.
        for(int i = 0; i < NUM_BLOCKS; i++){
            glm::vec4 q(blocks[i][0], blocks[i][1], blocks[i][2], 1);
            q = _mvpShader->M()*q;
            float d = hypot(q.x-p.x, q.y-p.y);
            if(d<=threshold){
                playerPos[0] = blocks[i][0];
                playerPos[1] = blocks[i][1];
                playerPos[2] = blocks[i][2];
                if(antigravity) playerPos[2]--;
                else            playerPos[1]++;
                return i;
            }
        }
        if(antigravity) playerPos[2]--;
        else            playerPos[1]++;
        return -1;
    }

    virtual void update(){
        //The direction of the camera. One of four states.
        //Each state rotates the world 90 degrees.
        //Incrementing direction rotates the camera counter clockwise and the world counter clockwise.
        static int direction = 0;
        //A LUT that can be used to map the game direction to a camera direction
        const int cameraAngles[4][3] = {
            {-1, -1, -1},
            {-1, -1,  1},
            { 1, -1,  1},
            { 1, -1, -1}
        };
        
        static int player[3] = {0, 0, 0};

        static bool keyInput[4];     //previous keyboard right, up, left, down
        static bool currKeyInput[4]; //Keyboard right, up, left, down
        //Player behavior
        if(_renderer != NULL && _player != NULL) {
            //Register keyboard inputs
            currKeyInput[0] = glfwGetKey(_renderer->window(), GLFW_KEY_W) == GLFW_PRESS;
            currKeyInput[1] = glfwGetKey(_renderer->window(), GLFW_KEY_Q) == GLFW_PRESS;
            currKeyInput[2] = glfwGetKey(_renderer->window(), GLFW_KEY_A) == GLFW_PRESS;
            currKeyInput[3] = glfwGetKey(_renderer->window(), GLFW_KEY_S) == GLFW_PRESS;
            int keyDirection = -1;
            //Move the player based on keyboard input
            if(currKeyInput[0] && !keyInput[0]) keyDirection = 0;
            if(currKeyInput[1] && !keyInput[1]) keyDirection = 1;
            if(currKeyInput[2] && !keyInput[2]) keyDirection = 2;
            if(currKeyInput[3] && !keyInput[3]) keyDirection = 3;
            if(keyDirection>=0){
                //Try moving the player
                move(player, direction, keyDirection);
                //Snap the player's position to a block
                int snappedBlockID = snap(player);
                if(snappedBlockID>=0){
                    printf("Snapped to %d at (%d, %d, %d)\n", snappedBlockID, blocks[snappedBlockID][0], blocks[snappedBlockID][1], blocks[snappedBlockID][2]);
                    switch(blocks[snappedBlockID][3]){
                        case BlockType::ROTATE_LEFT:
                            direction++;
                            if(direction>3) direction=0;
                        case BlockType::ROTATE_RIGHT:
                            direction--;
                            if(direction<0) direction=3;
                    }
                }
                else{ //No block to snap to; move back
                    move(player, direction, keyDirection, true);
                }

            }
            //Update the actual player's position
            float delta = _deltaTime*PLAYER_TRANSLATION_SPEED;
            _player->_x = (_player->_x+player[0]*delta)/(1+delta);
            _player->_y = (_player->_y+player[1]*delta)/(1+delta);
            _player->_z = (_player->_z+player[2]*delta)/(1+delta);
            //Temporary: Rotate the camera with the letter "r"
            if(glfwGetKey(_renderer->window(), GLFW_KEY_R) == GLFW_PRESS) {
                direction = (direction + 1)%4;
            }
            //Update registered keyboard locks
            for(int i = 0; i < 4; i++) keyInput[i] = currKeyInput[i];
            //Update the camera position
            delta = _deltaTime*CAMERA_TRANSLATION_SPEED;
            _mvpShader->cameraPosition[0] = (_mvpShader->cameraPosition[0]+(_player->_x-5*_mvpShader->cameraDirection[0])*delta)/(1+delta);
            _mvpShader->cameraPosition[1] = (_mvpShader->cameraPosition[1]+(_player->_y-5*_mvpShader->cameraDirection[1])*delta)/(1+delta);
            _mvpShader->cameraPosition[2] = (_mvpShader->cameraPosition[2]+(_player->_z-5*_mvpShader->cameraDirection[2])*delta)/(1+delta);
            //Update the camera direction
            delta = _deltaTime*CAMERA_ROTATION_SPEED;
            _mvpShader->cameraDirection[0] = (_mvpShader->cameraDirection[0]+cameraAngles[direction][0]*delta)/(1+delta);
            _mvpShader->cameraDirection[1] = (_mvpShader->cameraDirection[1]+cameraAngles[direction][1]*delta)/(1+delta);
            _mvpShader->cameraDirection[2] = (_mvpShader->cameraDirection[2]+cameraAngles[direction][2]*delta)/(1+delta);
            
        }
        World::update();
    }

    private:
        Shaders::TexturedMVP *_mvpShader;
        ObjFileObject *_player;
};