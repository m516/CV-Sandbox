#pragma once
#include "ray.h"

struct HitRecord {
    Point3 p;
    Vec3 normal;
    double t;
    bool frontFace;

    inline void setFaceNormal(const Ray& r, const Vec3& outward_normal) {
        frontFace = dot(r.direction(), outward_normal) < 0;
        normal = frontFace ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec) = 0;
};
