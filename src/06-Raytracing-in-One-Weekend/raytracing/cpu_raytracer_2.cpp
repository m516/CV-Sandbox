#include "cpu_raytracer_2.h"

void sliderDouble(const char* label, double* v) {
	float f = (float)*v;
	ImGui::DragFloat(label, &f);
	*v = f;
}

void displayVec(Vec3* v) {
	char data[24];
	sprintf(data, "(%.2f, %.2f, %.2f)", v->x(), v->y(), v->z());
	ImGui::Text(data);
}

double hitSphere(const Point3& center, double radius, const Ray& r) {
	Vec3 oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0) {
		return -1.0;
	}
	else {
		return (-half_b - sqrt(discriminant)) / a;
	}
}


void sliderVec3(const char* label, Vec3* v) {
	ImGui::Text(label);
	size_t i = strlen(label) + 1;
	char* newLabel = new char[i+1];
	strcpy(newLabel, label);
	strcat(newLabel, " x");
	sliderDouble(newLabel, &(v->e[0]));
	newLabel[i] = 'y';
	sliderDouble(newLabel, &(v->e[1]));
	newLabel[i] = 'z';
	sliderDouble(newLabel, &(v->e[2]));
}

void CPURaytracer2::addCustomUI()
{
	ImGui::Text("Camera");
	sliderVec3("Position", &(camera.orig));
	bool dirChanged = false;
	static float pitch = 0, yaw = 0, roll = 0;
	if(ImGui::DragFloat("Camera pitch", &pitch)) dirChanged = true;
	if (ImGui::DragFloat("Camera yaw", &yaw)) dirChanged = true;
	if (ImGui::DragFloat("Camera roll", &roll)) dirChanged = true;
	if (dirChanged) {
		Vec3 out(0, 1, 0);
		Vec3 up(0, 0, 1);
		RotationMatrix r;
		r.setRotation(pitch, yaw, roll, true);
		r.applyRotation(&out, &camera.dir);
		r.applyRotation(&up, &camera.up);
	}
	sliderDouble("Focal length", &(camera.focalLength));

	ImGui::Separator();
	ImGui::Text("Camera out");
	displayVec(&camera.dir);
	ImGui::Text("Camera up");
	displayVec(&camera.up);

	ImGui::Separator();
	ImGui::Text("Sampling");
	ImGui::SliderInt("Number of samples", &numSamples, 1, 50);
	ImGui::SliderInt("Maximum bounces per sample", &maxBounces, 1, 50);


}

inline void gammaCorrect(Color* c) {
	c->e[0] = sqrt(c->e[0]);
	c->e[1] = sqrt(c->e[1]);
	c->e[2] = sqrt(c->e[2]);
}

Color CPURaytracer2::colorAt(float x, float y, float aspectRatio)
{
	//Force center to be (0,0)
	x = 2 * x - 1;
	y = -2 * y + 1;
	
	//Force aspect ratio
	y /= aspectRatio;

	Vec2 imageCoords(x, y);

	Color c;
	for (int i = 0; i < numSamples; i++) {
		c += rayColor(camera.computeCameraRay(imageCoords));
	}
	c = c / numSamples;
	gammaCorrect(&c);
	return c;
}

Color CPURaytracer2::rayColor(const Ray& r, int numBounces)
{
	if (numBounces > maxBounces) return Color(0,0,0);

	HitRecord hr;
	if (world.hit(r, .001, 1000, hr)) {
		Point3 target = hr.p + hr.normal + randomVec3InUnitSphere();
		return 0.5 * rayColor(Ray(hr.p, target - hr.p), numBounces + 1);
	}

	Vec3 normalizedDirection = unitVector(r.direction());
	double t = 0.5 * (normalizedDirection.z() + 1.0);
	return (1.0 - t) * Color(.22, .2, .1) + t * Color(0, 0.3, 0.64);
}
