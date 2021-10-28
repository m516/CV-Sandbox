#pragma once

#include <vector>
#include <string>
#include <array>
#include "happly/happly.h"

namespace PointCloud {
	
	template<std::size_t D>	
	class Data: public std::vector<std::array<float, D>> {

	public:
		Data(size_t initial_size) :std::vector<std::array<float, D>>(initial_size) {};
		Data() {};
	};


	Data<6> loadXYZRGB(std::string filename);
}