#pragma once

#include <memory>

#include "icp.hpp"
#include "point.hpp"
#include "pointcloud.hpp"

namespace PointCloud {
	template<std::size_t D>
	class KDTree {
	public:
		KDTree(const Data<D>& pointCloud);
	private:
		Point<D> point;
		std::unique_ptr<KDTree<D>> left, right;
	};
}