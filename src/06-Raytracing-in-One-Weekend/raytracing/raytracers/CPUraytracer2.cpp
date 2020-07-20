#include "CPURaytracer2.h"

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

void CPURaytracer2::init()
{
	camera.dir = Vec3(0, 1, 0);
	camera.up = Vec3(0, 0, 1);
	camera.orig = Vec3(0, 0, 0);
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

}


Color rayColor(const Ray& r) {
	Vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.z() + 1.0);
	return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.3, 0.5, 1.0);
}


Color CPURaytracer2::colorAt(float x, float y, float aspectRatio)
{
	//Force center to be (0,0)
	x = 2 * x - 1;
	y = -2 * y + 1;
	
	//Force aspect ratio
	y /= aspectRatio;

	Vec2 imageCoords(x, y);

	return rayColor(camera.computeCameraRay(imageCoords));
}
