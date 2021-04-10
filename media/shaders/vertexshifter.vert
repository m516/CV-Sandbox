#version 330 core

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 color;

out vec4 vColor;


uniform float xoffset;
uniform float yoffset;
uniform float xscale;
uniform float yscale;


void main() {
	vec4 Position = vec4(vPosition, 1.0);
	Position.x *= xscale;
	Position.y *= yscale;
	Position.x += xoffset;
	Position.y += yoffset;	

	gl_Position =  Position;
	vColor = vec4(color, 1.0);
}

