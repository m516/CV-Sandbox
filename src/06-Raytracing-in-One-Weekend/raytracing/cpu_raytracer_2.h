#pragma once
#include <string>
#include "generic_cpu_raytracer.h"
#include "raytracing_utils.h"
#include "util/vec3.h"
#include "util/vec2.h"
#include "util/camera.h"
#include "util/rot_math.h"
#include "util/hittable_list.h"
#include "util/primitives/sphere.h"

//Renders a normals-Colored sphere in the middle of the image and 
class CPURaytracer2 : GenericCPUTracer{
public:
	CPURaytracer2() {
		shortDescription = "CPU Raytracer 02: A Circle";
		longDescription = "This simple algorithm tests Rays hitting spheres.";

		camera.dir = Vec3(0, 1, 0);
		camera.up = Vec3(0, 0, 1);
		camera.orig = Vec3(0, 0, 0);


		//Add a couple spheres to the world
		for (int i = 0; i < 50; i++) {
			double radius = randomDouble(0, 10);
			Vec3 pos (randomDouble(-50, 50), randomDouble(-50, 50), radius);
			world.add(make_shared<Sphere>(pos, radius));
		}

		//world.add(make_shared<Sphere>(Vec3(0, 50, 0), 25));
	}
	virtual void addCustomUI();
	virtual Color colorAt(float x, float y, float aspectRatio);
private:
    // Camera
	Camera camera;
	int numSamples = 1;
	int maxBounces = 2;
	HittableList world;
	Color rayColor(const Ray& r, int numBounces = 0);
};