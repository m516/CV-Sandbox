/* Hello GLFW
*  Based on https://www.glfw.org/docs/3.3/quick.html
*/

#include "main.hpp"
#include <thread>

#include "happly/happly.h"

#include "pointcloud/pointcloud.hpp"
#include "pointcloud/renderer.hpp"
#include "app/window.hpp"

using namespace std;



/**
 * A helper function for terminating the program
 */
void terminate(int errorCode) {
	using namespace std;
	cout << "Closing application";
	//Exit
	exit(errorCode);
}




int main()
{

	App::Window window(1080, 1080);
	if (!window.open()) terminate(1);

	//Initialize shaders
	PointCloud::Data<6> pointCloud = PointCloud::loadXYZRGB(MEDIA_DIRECTORY "german-shepherd-pointcloud.ply");
	PointCloud::Renderer renderer;
	renderer.set(pointCloud);



	//The render loop
	while (!window.shouldClose())
	{
		window.clear();



		float cameraPosition[] = { -2, 1, 2 };
		float cameraCenter[] = { 0, .5, 0 };
		renderer.setViewLookAt(cameraPosition, cameraCenter);


		renderer.display();

		window.refresh();

		using namespace std::this_thread;
		using namespace std::chrono;
		sleep_for(milliseconds(20));
	}

	window.close();
}

