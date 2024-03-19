#ifndef __GAUSSIAN_OCTREE__
#define __GAUSSIAN_OCTREE__

#pragma once
#include "PLYReader.h"

class GaussianOctree
{
private:
    GaussianOctree* children = {nullptr};
    uint32_t* containedSplats = nullptr;
    uint8_t level = 0;

public:
    GaussianOctree(uint32_t * splatArray, uint32_t numPrimitives, uint8_t _level, SplatData * sd, glm::vec3 * bbox);
    ~GaussianOctree();
};

GaussianOctree::GaussianOctree(uint32_t * splatArray, uint32_t numPrimitives, uint8_t _level, SplatData * sd, glm::vec3 * bbox)
{
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