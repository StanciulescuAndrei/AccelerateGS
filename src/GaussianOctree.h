#ifndef __GAUSSIAN_OCTREE__
#define __GAUSSIAN_OCTREE__

#define MAX_OCTREE_LEVEL 6

#pragma once
#include "PLYReader.h"
#include <vector>

class GaussianOctree
{
private:
    GaussianOctree* children[8] = {nullptr};
    std::vector<uint32_t> containedSplats = nullptr;
    uint8_t level = 0;
    bool isLeaf = false;
    glm::vec3 bbox[2];

public:
    void processSplats(uint32_t * splatArray, uint32_t numPrimitives, uint8_t _level, SplatData * sd); 
    GaussianOctree( glm::vec3 * _bbox);
    ~GaussianOctree();
};

GaussianOctree::GaussianOctree(glm::vec3 * _bbox)
{
    bbox[0] = _bbox[0];
    bbox[1] = _bbox[1];
}

void GaussianOctree::processSplats(uint32_t * splatArray, uint32_t numPrimitives, uint8_t _level, SplatData * sd){
    if(numPrimitives == 0){
        return;
    }

    memcpy(containedSplats, splatArray, sizeof(uint32_t) * numPrimitives);

    if(numPrimitives < 8) // come threshold where it's not worth going deeper
        return;

    glm::vec3 halfSize = (bbox[1] - bbox[0]) * 0.5f;

    for (int i=0;i<8;i++){
        children[i] = new OctreeNode();
        // Define node's BBox
        glm::vec3 childBbox[2];
        childBbox[0][0] = bbox[0][0] + ((i & 0b001)!=0) * halfSize[0];
        childBbox[0][1] = bbox[0][1] + ((i & 0b010)!=0) * halfSize[1];
        childBbox[0][2] = bbox[0][2] + ((i & 0b100)!=0) * halfSize[2];
        childBbox[1] = childBbox[0] + halfSize;
        // See which of the vertices go into the newly created node
        // for(auto splat : containedSplats){
        //     if(insideBBox(currentNode->children[i]->bbox, vertices->at(v))){
        //         currentNode->children[i]->verts_id.push_back(v);
        //     }
        // }
        // if(crt_depth < max_depth){
        //     currentNode->children[i]->isLeaf = false;
        //     processNode(currentNode->children[i], vertices, crt_depth+1, max_depth);
        // }
        // else
        //     currentNode->children[i]->isLeaf=true;
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