#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 texCoord;

out vec4 vPos;
out vec4 vColor;
out vec4 vTexCoord;

void main() {
  vPos = vec4(position, 1.0);
  gl_Position = vPos;
  vColor = vec4(color, 0.0);
  vTexCoord = texCoord;
}