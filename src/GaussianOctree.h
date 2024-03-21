#ifndef __GAUSSIAN_OCTREE__
#define __GAUSSIAN_OCTREE__

#define MAX_OCTREE_LEVEL 6

#pragma once
#include "PLYReader.h"
#include <vector>

bool insideBBox(glm::vec3 * bbox, uint32_t splatId, SplatData * sd){
    float maxRadius = max(sd[splatId].fields.scale[0], max(sd[splatId].fields.scale[1], sd[splatId].fields.scale[2]));
    glm::vec3 minBound = glm::make_vec3(sd[splatId].fields.position) - maxRadius;
    glm::vec3 maxBound = glm::make_vec3(sd[splatId].fields.position) + maxRadius;

    if(minBound.x >= bbox[0].x && minBound.y >= bbox[0].y && minBound.z >= bbox[0].z &&
       maxBound.x < bbox[1].x && maxBound.y < bbox[1].y && maxBound.z < bbox[1].z)
       return true;

    return false;
}

class GaussianOctree
{
public:
    GaussianOctree* children[8] = {nullptr};
    std::vector<uint32_t> containedSplats = nullptr;
    uint8_t level = 0;
    bool isLeaf = false;
    glm::vec3 bbox[2];


    void processSplats(uint8_t _level, SplatData * sd); 
    GaussianOctree( glm::vec3 * _bbox);
    ~GaussianOctree();
};

GaussianOctree::GaussianOctree(glm::vec3 * _bbox)
{
    bbox[0] = _bbox[0];
    bbox[1] = _bbox[1];
}

void GaussianOctree::processSplats(uint8_t _level, SplatData * sd){
    if(containedSplats.size() == 0){
        isLeaf = true;
        return;
    }

    if(containedSplats.size() < 8){ // Some threshold where it's not worth going deeper
        isLeaf = true;
        return;
    }

    glm::vec3 halfSize = (bbox[1] - bbox[0]) * 0.5f;

    for (int i=0;i<8;i++){
        // Define node's BBox
        glm::vec3 childBbox[2];
        childBbox[0][0] = bbox[0][0] + ((i & 0b001)!=0) * halfSize[0];
        childBbox[0][1] = bbox[0][1] + ((i & 0b010)!=0) * halfSize[1];
        childBbox[0][2] = bbox[0][2] + ((i & 0b100)!=0) * halfSize[2];
        childBbox[1] = childBbox[0] + halfSize;

        children[i] = new GaussianOctree(childBbox);

        // See which of the splats go into the newly created node
        for(auto splat : containedSplats){
            if(insideBBox(children[i]->bbox, splat, sd)){
                children[i]->containedSplats.push_back(splat);
            }
        }
        if(level < MAX_OCTREE_LEVEL){
            children[i]->isLeaf = false;
            children[i]->processSplats(_level+1, sd);
        }
        else
            currentNode->children[i]->isLeaf=true;
    }

}

GaussianOctree::~GaussianOctree()
{
    if(containedSplats != nullptr){
        delete [] containedSplats; 
    }
    if(children != nullptr){
        delete [] children;
    }
}


#endif