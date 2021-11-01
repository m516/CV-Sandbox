#version 330 core
// Takes in input 


layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;


out vec4 vPos;
out vec4 vColor;

uniform mat4 TransformMatrix; // Transforms the point cloud. 

void main() {
  vPos = vec4(position, 1.0);
  vPos = TransformMatrix * vPos;
  gl_Position = vPos;
  vColor = vec4(color, 0.0);
}

