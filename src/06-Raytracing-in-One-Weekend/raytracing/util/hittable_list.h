#pragma once

#include "hittable.h"

#include <memory>
#include <vector>

using std::shared_ptr;
using std::make_shared;

class HittableList : public Hittable {
public:
    HittableList() {}
    HittableList(shared_ptr<Hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(shared_ptr<Hittable> object) { objects.push_back(object); }

    virtual bool hit(const Ray& r, double tMin, double tMax, HitRecord& rec);

public:
    std::vector<shared_ptr<Hittable>> objects;
};


