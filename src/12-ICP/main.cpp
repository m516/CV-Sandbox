/* Hello GLFW
*  Based on https://www.glfw.org/docs/3.3/quick.html
*/

#include "main.hpp"

#include "happly/happly.h"

#include "pointcloud/pointcloud.hpp"
#include "pointcloud/icp.hpp"
// #include "pointcloud/kdtree.hpp"
#include "app/window.hpp"
#include "app/GUI.hpp"
using namespace std;





int main()
{
	using namespace App::UI;



	//PointCloud::Data<6> p1 = PointCloud::loadXYZRGB("C:/Users/Beta/Documents/school/ME456/project/mesh1.ply");
	//PointCloud::KDTree<6> tree(p1);
	//PointCloud::Data<6> p2 = PointCloud::loadXYZRGB("C:/Users/Beta/Documents/school/ME456/project/mesh2.ply");
	// PointCloud::ICP<6>(p1, p1);
	// PointCloud::ICP<6>(p1, p2);
	// PointCloud::ICP<6>(p2, p2);

	//Create a new UI
	init();
	
	//Initialize the point cloud renderer
	//setPointCloud(MEDIA_DIRECTORY "german-shepherd-pointcloud.ply");
	setPointCloud("C:/Users/Beta/Documents/school/ME456/project/data/WVCRoomScanningPointCloud/T1.ply");
	
	//Run the application
	run();
}

