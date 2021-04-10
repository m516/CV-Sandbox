#version 330 core

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vNormal;
out vec4 vColor;

uniform vec4 AmbientProduct, DiffuseProduct, SpecularProduct;
uniform mat4 ModelView;
uniform mat4 Projection;
uniform vec4 LightPosition;

void main(){
	vec4 ambient, diffuse, specular;
	float Shininess = 50.0;
	float Kd, Ks;

	// Transform position and normal into eye coordinates
	vec4 V = vec4(vPosition, 1);
	vec3 pos = (ModelView * V).xyz;
	vec4 NN = ModelView * vec4(vNormal, 0.0);
	vec3 N = normalize(NN.xyz);

	// Output position
	gl_Position =  Projection * ModelView * V;

	vec3 L = normalize(LightPosition.xyz - pos);
	vec3 E = normalize(-pos);
	vec3 H = normalize(L+E);

	// Compute the illumination equation
	ambient = AmbientProduct;
	Kd = max(dot(L, N), 0.0);
	diffuse = Kd * DiffuseProduct;

	Ks = pow(max(dot(N, H), 0.0), Shininess);
	specular = max(pow(max(dot(N, H), 0.0), Shininess) * Ks * SpecularProduct, 0.0);

	vColor = vec4((ambient + diffuse + specular).xyz, 1.0);

}

