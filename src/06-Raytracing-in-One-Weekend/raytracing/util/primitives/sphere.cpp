#include "sphere.h"

bool Sphere::hit(const Ray& r, double t_min, double t_max, HitRecord& rec)
{
    Vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant > 0) {
        auto root = sqrt(discriminant);

        auto temp = (-half_b - root) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            Vec3 outward_normal = (rec.p - center) / radius;
            rec.setFaceNormal(r, outward_normal);
            return true;
        }

        temp = (-half_b + root) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            Vec3 outward_normal = (rec.p - center) / radius;
            rec.setFaceNormal(r, outward_normal);
            return true;
        }
    }

    return false;
}
