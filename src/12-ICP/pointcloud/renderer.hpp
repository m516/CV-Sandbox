#pragma once

#include <memory>
#include <vector>
#include "pointcloud.hpp"
#include "glad/glad.h"
#include "glm/glm.hpp"

namespace PointCloud {
	class Renderer {
	public:
		void set(Data<6> xyzrgb);
		void setSparse(Data<6> xyzrgb, double percent);
		void display();
		Renderer() {}

		void setViewLookAt(float cameraPosition[3], float centerOfFocus[3]);

		static glm::mat4 viewMatrix;
		glm::mat4 transformMatrix;
	private:
		static bool initialized;
		static void initialize();
		static GLuint shaderID;
		static GLuint uniformViewID, uniformTransformID;
		static GLuint vertexArrayID, positionBufferID, colorBufferID;
		size_t size;
	};
}