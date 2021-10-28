#pragma once

#include <vector>

#include "../pointcloud/pointcloud.hpp"
#include "../pointcloud/renderer.hpp"
#include "window.hpp"


namespace App {
	namespace UI {
		void init();
		void setPointCloud(std::string filename);
		void run();
	};
}