#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texCoord;

out vec3 vPos;
out vec3 vColor;
out vec2 vTexCoord;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;

void main(){

	// Output position of the vertex, in clip space : MVP * position
	gl_Position =  MVP * vec4(vertexPosition_modelspace, 1);
	vPos = vertexPosition_modelspace;
	vColor = color;
  	vTexCoord = texCoord;

}

