#pragma once

#include <vector>
#include <string>
#include <array>
#include "happly/happly.h"
#include "point.hpp"

namespace PointCloud {
	
	template<std::size_t D>	
	class Data: public std::vector<Point<D>> {

	public:
		Data(size_t initial_size) :std::vector<Point<D>>(initial_size) {};
		Data() {};
	};


	Data<6> loadXYZRGB(std::string filename);
}