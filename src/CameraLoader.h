#ifndef __CAMERA_LOADER__
#define __CAMERA_LOADER__

#include <fstream>
#include <iostream>
#include "../../../libs/nlohmann/json.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

nlohmann::json cameraData;

void loadCameraFile(std::string c){
    std::ifstream file(c);

    if(!file){
        std::cout<<"Unable to open camera file: "<<c<<std::endl;
    }

    file >> cameraData;
    std::cout << "Camera positions loaded: " << cameraData.size() << std::endl;
}

void loadGenericProperties(int & screen_x, int & screen_y, float & fovx, float & fovy){
    auto element = cameraData[0];

    screen_x = element["width"].get<int>();
    screen_y = element["height"].get<int>();

    int sensorx = element["fx"].get<int>();
    int sensory = element["fy"].get<int>();

    fovx = 2.0f * atanf((float)screen_x / (2.0f * sensorx));
    fovy = 2.0f * atanf((float)screen_y / (2.0f * sensory));
}

void getCameraParameters(int idx, glm::vec3 & position, glm::mat3 & rotation){
    if(idx >= cameraData.size()){
        std::cout<<"Invalid camera position!"<<std::endl;
        return;
    }

    auto element = cameraData[idx];
    std::vector<float> data;
    data = element["position"].get<std::vector<float>>();
    position = glm::make_vec3(&data[0]);
    for(int i=0;i < 3; i++){
        data = element["rotation"][i].get<std::vector<float>>();
        rotation[0][i] = data[0];
        rotation[1][i] = data[1];
        rotation[2][i] = data[2];
    }

}

#endif