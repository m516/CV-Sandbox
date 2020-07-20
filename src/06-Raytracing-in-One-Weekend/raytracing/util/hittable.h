#include "ray.h"

struct hit_record {
    Point3 p;
    Vec3 normal;
    double t;
};

class hittable {
public:
    virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};
