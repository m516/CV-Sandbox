#pragma once
#include "../ray.h"
#include "../hittable.h"

class Sphere : public Hittable {
public:
	Point3 center;
	double radius = 1;

	Sphere() {}
	Sphere(Point3 center, double radius) { this->center = center; this->radius = radius; }

	virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec);
};