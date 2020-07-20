#include "CPUraytracer1.h"

void CPURaytracer1::addCustomUI()
{
	GenericCPUTracer::addCustomUI();
	ImGui::Text("For this colored box, you just have to press the 'render' button.");
}

Color CPURaytracer1::colorAt(float x, float y, float aspectRatio)
{
	Color c(x, y, 1. - y);
	return c;
}
