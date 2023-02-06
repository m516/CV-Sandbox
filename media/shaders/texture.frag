#version 330 core
in vec3 vPos;
in vec3 vColor;
in vec2 vTexCoord;

uniform sampler2D blockTexture;

out vec4 fColor;

void main(void) {
  //vec4 t = vColor;
  //t = t * vPos.y;
  fColor = texture(blockTexture, vTexCoord) * vec4(vColor, 1);
}