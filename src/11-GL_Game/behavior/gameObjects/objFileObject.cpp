#include "objFileObject.h"

/**
 * @brief A utility for loading 
 * 
 * @param path 
 * @param out_vertices 
 * @param out_normals 
 * @return true object loading was successful
 * @return false object loading failed.
 */
bool loadOBJ2 (const char * path, std::vector<glm::vec3> & out_vertices, std::vector<glm::vec2> & out_uvs, std::vector<glm::vec3> & out_normals) {
	printf("Loading OBJ file %s...\n", path);

	std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
	std::vector<glm::vec3> temp_vertices; 
	std::vector<glm::vec2> temp_uvs; 
	std::vector<glm::vec3> temp_normals;

	FILE * file = fopen(path, "r");
	if( file == NULL ){
		printf("File %s not found\n", path);
		return false;
	}

	while(true) {

		char lineHeader[128];
		// read the first word of the line
		int res = fscanf(file, "%s", lineHeader);
		if (res == EOF)
			break; // EOF = End Of File. Quit the loop.

		// else : parse lineHeader
		int temp;
        if ( strcmp( lineHeader, "v" ) == 0 ){
            glm::vec3 vertex;
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
            temp_vertices.push_back(vertex);
		} else if ( strcmp( lineHeader, "vt" ) == 0 ){
			glm::vec2 uv;
			temp = fscanf(file, "%f %f\n", &uv.x, &uv.y );
			uv.y = 1-uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
			temp_uvs.push_back(uv);
		} else if ( strcmp( lineHeader, "vn" ) == 0 ){
			glm::vec3 normal;
			temp = fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z );
			temp_normals.push_back(normal);
        }else if ( strcmp( lineHeader, "f" ) == 0 ){
            std::string vertex1, vertex2, vertex3;
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
            if (matches != 9){
                printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                return false;
            }
            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            uvIndices    .push_back(uvIndex[0]);
            uvIndices    .push_back(uvIndex[1]);
            uvIndices    .push_back(uvIndex[2]);
            normalIndices.push_back(normalIndex[0]);
            normalIndices.push_back(normalIndex[1]);
            normalIndices.push_back(normalIndex[2]);
		} else{
			// Probably a comment, eat up the rest of the line
			char stupidBuffer[1000];
			char *temp;
			temp = fgets(stupidBuffer, 1000, file);
		}

	}

	// For each vertex of each triangle
	for(unsigned int i=0; i < vertexIndices.size(); i++ ) {

		// Get the indices of its attributes
		unsigned int vertexIndex = vertexIndices[i];
		unsigned int uvIndex     = uvIndices[i];
		unsigned int normalIndex = normalIndices[i];
		
		// Get the attributes thanks to the index
		glm::vec3 vertex = temp_vertices[ vertexIndex-1 ];
		glm::vec2 uv     = temp_uvs     [ uvIndex-1     ];
		glm::vec3 normal = temp_normals [ normalIndex-1 ];
		
		// Put the attributes in buffers
		out_vertices.push_back(vertex);
		out_uvs.push_back(uv);
		out_normals .push_back(normal);
	
	}
	fclose(file);
	return true;
}

ObjFileObject::ObjFileObject(const char* filename) : GameObject(){
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec3> normals;

    string f = MEDIA_DIRECTORY;
    f += filename;

    bool loaded = loadOBJ2(f.c_str(), vertices, uvs, normals);
    if(!loaded) RUNTIME_ERROR("-> Failed to load object file %s\n", f.c_str());

    _ownsVertices = true;
    _numVertices     = vertices.size();
    _vertexPositions = new float[_numVertices*3];
    _vertexUVs       = new float[_numVertices*2];
    _vertexNormals   = new float[_numVertices*3];
    _vertexColors    = new float[_numVertices*3];

    for(size_t v = 0; v < _numVertices; v++){
        _vertexPositions[v*3+0]=vertices[v][0];
        _vertexPositions[v*3+1]=vertices[v][1];
        _vertexPositions[v*3+2]=vertices[v][2];
        _vertexColors[v*3+0]=1;
        _vertexColors[v*3+1]=1;
        _vertexColors[v*3+2]=1;
        _vertexUVs[v*2+0]=uvs[v][0];
        _vertexUVs[v*2+1]=uvs[v][1];
    }
    printf("-> Load successful. Number of vertices: %zu\n", _numVertices);
}

void ObjFileObject::update(){
    for(int i = 0; i < _numVertices; i++){
        for(int j = 0; j < 3; j++){
            _vertexArray[i]->position[j]=_vertexPositions[i*3+j]+_pos[j];
            _vertexArray[i]->color[j]   =_vertexColors[i*3+j]*_col[j];
        }
        for(int j = 0; j < 2; j++){
            _vertexArray[i]->uv[j]      =_vertexUVs[i*2+j];
        }
    }
}