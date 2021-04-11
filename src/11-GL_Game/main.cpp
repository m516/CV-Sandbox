#include "main.h"
#include "engine/renderer.h"
#include "behavior/worlds/w1_triangle.h"
#include "behavior/worlds/w3_objects.h"

#define STB_IMAGE_IMPLEMENTATION
#include "engine/stb_image.h"

using namespace std;




int main()
{
	Renderer r;
    World3_Objects w(r.vertexArray);
    w.update();

    cout << "Media directory: " << MEDIA_DIRECTORY << endl;
    r.init(&w);

    while(true){
        w.update();
        r.render();
    }
}
