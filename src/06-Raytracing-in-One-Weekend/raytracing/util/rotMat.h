#pragma once
#include "../raytracingutils.h"

class RotationMatrix {
public:
	double mat[3][3];
	void setRotation(double pitch, double yaw, double roll, bool degrees = false) {
		if (degrees) {
			pitch *= 0.01745329251994329576923690768489;
			yaw *= 0.01745329251994329576923690768489;
			roll *= 0.01745329251994329576923690768489;
		}
		double sinA = sin(yaw),
			sinB = sin(roll),
			sinC = sin(pitch),
			cosA = cos(yaw),
			cosB = cos(roll),
			cosC = cos(pitch);
			;
		mat[0][0] = cosA * cosB;  mat[1][0] = cosA * sinB * sinC - sinA * cosC;  mat[2][0] = cosA * sinB * cosC + sinA * sinC;
		mat[0][1] = sinA * cosB;  mat[1][1] = sinA * sinB * sinC + cosA * cosC;  mat[2][1] = cosA * sinB * cosC - cosA * sinC;
		mat[0][2] = -sinB;        mat[1][2] = cosB * sinC;                       mat[2][2] = cosB * cosC;
	}

	void applyRotation(Vec3* v, Vec3* out) {
		out->e[0] = mat[0][0] * v->e[0] + mat[1][0] * v->e[1] + mat[2][0] * v->e[2];
		out->e[1] = mat[0][1] * v->e[0] + mat[1][1] * v->e[1] + mat[2][1] * v->e[2];
		out->e[2] = mat[0][2] * v->e[0] + mat[1][2] * v->e[1] + mat[2][2] * v->e[2];
	}
};