#include "kdtree.hpp"
#include <random>

using namespace PointCloud;


std::size_t rand_range(std::size_t min, std::size_t max) {
	static std::random_device                  rand_dev;
	static std::mt19937                        generator(rand_dev());
	std::uniform_int_distribution<size_t>  distr(min, max);
	return distr(generator);
}

template<std::size_t D> 
bool comparePoints(const PointCloud::Point<D> &p1, const PointCloud::Point<D> &p2) {
	return false; //TODO
}

template<std::size_t D> PointCloud::KDTree<D>::KDTree(const PointCloud::Data<D>& pointCloud) {
	PointCloud::Data<D> midPointSearchVector(25);
	if (pointCloud.size() > 100) {
		for (int i = 0; i < 25; i++) {
			//Grab a random index
			int randomIndex = rand_range(0,pointCloud.size()); //TODO
			midPointSearchVector.push_back(pointCloud[randomIndex]);
		}
	}
	else {
		midPointSearchVector = pointCloud;
	}

}
