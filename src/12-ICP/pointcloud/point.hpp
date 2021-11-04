#pragma once

#include <array>
#include <memory>
#include <cmath>

namespace PointCloud {
	template<size_t D>
	class Point : public std::array<float, D> {
	public:
		float distanceSquaredTo(const Point<D> other) {
			float r = 0.f;
			for (int i = 0; i < D; i++) {
				float t = p2[0] - p1[0];
				r += t * t;
			}
			return r;
		}

		float distanceTo(const Point<D>& other) {
			return sqrtf(distanceSquaredTo(other));
		}
		
	};
}