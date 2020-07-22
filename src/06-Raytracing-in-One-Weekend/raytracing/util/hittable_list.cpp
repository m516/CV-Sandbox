#include "hittable_list.h"

bool HittableList::hit(const Ray& r, double tMin, double tMax, HitRecord& rec) {
    HitRecord tempRecord;
    bool hitAnything = false;
    auto closestHitSoFar = tMax;

    for (shared_ptr<Hittable> object : objects) {
        if (object->hit(r, tMin, closestHitSoFar, tempRecord)) {
            hitAnything = true;
            closestHitSoFar = tempRecord.t;
            rec = tempRecord;
        }
    }

    return hitAnything;
}