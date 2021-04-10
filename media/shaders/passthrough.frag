#version 330 core
in vec4 vPos;
in vec4 vColor;
out vec4 fColor;

void main(void) {
  vec4 t = vColor;
  //t = t * vPos.y;
  fColor = t;
}