#pragma once

#include "ray.h"


class Camera : public Ray{
public:
	Vec3 up = Vec3(0, 0, 1);
	double focalLength = 1.0;
	/*
	Input: coordinates in the image space, where (-1,-1) is the lower left-hand corner
	and (1,1) is the upper right-hand corner

	Output: a ray coming from the camera
	*/
	Ray computeCameraRay(Vec2 imageCoordinates) {
		Vec3 right = cross(up, dir);
		Ray r;
		r.orig = orig;
		r.dir = imageCoordinates.x() * right + imageCoordinates.y() * up + focalLength * dir ;
		return r;
	}
};


/*
An implementation of the camera matrix, used for transforming points and rays into camera space
See https://en.wikipedia.org/wiki/Camera_matrix#Normalized_camera_matrix_and_normalized_image_coordinates
*/
/*
class CameraMatrix {
public:
	double mat[4][3];
	//Initializes the transformation matrix
	void setMatrix(Point3 position, Vec3 rotation, float focalLength, bool degrees = false) {
		if (degrees) {
			degrees *= 0.01745329251994329576923690768489;
		}

		double sinA = sin(rotation.x()),
			sinB = sin(rotation.y()),
			sinC = sin(rotation.z()), 
			cosA = cos(rotation.x()),
			cosB = cos(rotation.y()),
			cosC = cos(rotation.z());
		mat[0][0] = cosA * cosB;  mat[1][0] = cosA * sinB * sinC - sinA * cosB;  mat[2][0] = cosA * sinB * cosC + sinA * sinC;
		mat[0][1] = sinA * cosB;  mat[1][1] = sinA * sinB * sinC + cosA * cosC;  mat[2][1] = cosA * sinB * cosC - cosA * sinC;
		mat[0][2] = -sinB;        mat[1][2] = cosB * sinB;                       mat[2][2] = cosB * cosC / focalLength;
	}

	void setPosition(Point3 newPosition) {
		//Position
		mat[3][0] = position.x();
		mat[3][1] = position.y();
		mat[3][2] = position.z();
	}

	Vec2 imageCoordinatesOf(Vec3 position) {
		Vec2 v;
		v.e[0]=position.x*mat[]
		return v;
	}

	Ray generateRay(Vec2 position) {

	}

};
*/