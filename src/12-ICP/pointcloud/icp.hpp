#pragma once
#include "pointcloud.hpp"
#include "glm/glm.hpp"


namespace PointCloud {



	template<std::size_t D>
	float distance(const std::array<float, D>& p1, const std::array<float, D>& p2) {
		float r = 0.f;
		for (int i = 0; i < D; i++) {
			float t = p2[0] - p1[0];
			r += t * t;
		}
		//r = sqrtf(r);
		return r;
	}

	// Simple algorithm to recursively find the closest point
	template<std::size_t D>
	float closestPoint(const PointCloud::Data<D>& pointCloud1, const PointCloud::Data<D>& pointCloud2,
			std::size_t low = ~0, std::size_t high = ~0) {

		if (low == ~0 && high == ~0) {
			low = 0;
			high = pointCloud1.size();
		}

		if (high - low > 30000) {
			std::size_t mid = (high + low) / 2;
			float d = 0.f;
			d += closestPoint<D>(pointCloud1, pointCloud2, low, mid);
			d += closestPoint<D>(pointCloud1, pointCloud2, mid, high);
			return d;
		}
		float r = 0;
		for (size_t i = low; i < high; i+=1000) {
			float minDistance = std::numeric_limits<float>::infinity();
			for (size_t j = 0; j < pointCloud2.size(); j++) {
				float d = distance<D>(pointCloud1[i], pointCloud2[j]);
				if (d < minDistance) minDistance = d;
			}
			r += minDistance;
		}
		return r;
	}


	template<std::size_t D>
	glm::mat4 ICP(const Data<D>& pointCloud1, const Data<D>& pointCloud2)
	{
		std::cout << "Closest point returns " << closestPoint<D>(pointCloud1, pointCloud2) << std::endl;

		glm::mat4 transformMatrix();

		return glm::mat4();
	}

}