/* Hello GLFW
*  Based on https://www.glfw.org/docs/3.3/quick.html
*/

#include "main.hpp"

#include "happly/happly.h"

#include "pointcloud/pointcloud.hpp"
#include "pointcloud/renderer.hpp"
#include "app/window.hpp"
#include "app/GUI.hpp"

using namespace std;





int main()
{
	using namespace App::UI;

	//Create a new UI
	init();

	//Initialize the point cloud renderer
	setPointCloud(MEDIA_DIRECTORY "german-shepherd.ply");

	//Run the application
	run();
}

