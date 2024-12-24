#ifndef __CAMERA_LOADER__
#define __CAMERA_LOADER__

#include <fstream>
#include <iostream>
#include "../../../libs/nlohmann/json.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "GUIManager.h"

nlohmann::json cameraData;

nlohmann::json appConfig;

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

    int sensorx = element["fx"].get<double>();
    int sensory = element["fy"].get<double>();

    fovx = 2.0f * atanf((float)screen_x / (2.0f * sensorx));
    fovy = 2.0f * atanf((float)screen_y / (2.0f * sensory));

    screen_x = 1920;
    screen_y = 1080;
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

void loadApplicationConfig(std::string c, RenderConfig & rc){
    std::ifstream file(c);

    if(!file){
        std::cout<<"Unable to open config file: "<<c<<std::endl;
    }

    file >> appConfig;
    rc.scene = appConfig["scene"].get<std::string>();
    rc.structure = appConfig["structure"].get<std::string>();
    rc.representative = appConfig["representative"].get<std::string>();
    rc.clustering = appConfig["clustering"].get<std::string>();
    rc.dbscan_epsilon = appConfig["dbscan_epsilon"].get<float>();
    rc.numClusterFeatures = appConfig["numClusterFeatures"].get<int>();
    rc.spectralClusteringThreshold = appConfig["spectralClusteringThreshold"].get<int>();
    rc.nClusters = appConfig["numClusters"].get<int>();
    rc.octreeLevel = appConfig["octreeLevel"].get<int>();
    std::cout << "Volumetric structure type: " << rc.structure << std::endl;
    std::cout << "# of required clusters: " << rc.nClusters << std::endl;
    std::cout << "Clustering type: " << rc.clustering << std::endl;
    std::cout << "Representative type: " << rc.representative << std::endl;
    std::cout << "Number of features for clustering: " << rc.numClusterFeatures << std::endl;
    std::cout << "Spectral cluster limit: " << rc.spectralClusteringThreshold << std::endl;
    std::cout << "Tree Octree depth: " << rc.octreeLevel << std::endl;
}

#endif