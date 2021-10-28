#include "pointcloud.hpp"

using namespace PointCloud;


Data<6> PointCloud::loadXYZRGB(std::string filename)
{
	happly::PLYData plyIn(filename);
	std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
	std::vector<std::array<unsigned char, 3>> vCol = plyIn.getVertexColors();
	Data<6> d(vPos.size());
	for (int i = 0; i < vPos.size(); i++) {
		d[i][0] = (float)vPos[i][0];
		d[i][1] = (float)vPos[i][1];
		d[i][2] = (float)vPos[i][2];
		d[i][3] = (float)vCol[i][0]/256.f;
		d[i][4] = (float)vCol[i][1]/256.f;
		d[i][5] = (float)vCol[i][2]/256.f;
	}
	return d;
}
