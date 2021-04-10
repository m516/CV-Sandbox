#include "main.h"
#include "engine/renderer.h"
#include "behavior/worlds/w1_triangle.h"
#include "behavior/worlds/w2_cube.h"

using namespace std;




int main()
{
	Renderer r;
    World2_CubeDemo w(r.vertexArray);
    w.update();

    cout << "Media directory: " << MEDIA_DIRECTORY << endl;
    r.init(&w);

    while(true){
        w.update();
        r.render();
    }
}
