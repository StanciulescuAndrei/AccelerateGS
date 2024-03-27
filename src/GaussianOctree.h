#ifndef __GAUSSIAN_OCTREE__
#define __GAUSSIAN_OCTREE__

#define MAX_OCTREE_LEVEL 12

#pragma once
#include "PLYReader.h"
#include <vector>

bool insideBBox(glm::vec3 * bbox, uint32_t splatId, std::vector<SplatData> & sd){
    // float maxRadius = max(sd[splatId].fields.scale[0], max(sd[splatId].fields.scale[1], sd[splatId].fields.scale[2]));
    // glm::vec3 minBound = glm::make_vec3(sd[splatId].fields.position) - maxRadius;
    // glm::vec3 maxBound = glm::make_vec3(sd[splatId].fields.position) + maxRadius;

    // if(minBound.x >= bbox[0].x && minBound.y >= bbox[0].y && minBound.z >= bbox[0].z &&
    //    maxBound.x < bbox[1].x && maxBound.y < bbox[1].y && maxBound.z < bbox[1].z)
    //    return true;

    if(sd[splatId].fields.position[0] >= bbox[0].x && sd[splatId].fields.position[1] >= bbox[0].y && sd[splatId].fields.position[2] >= bbox[0].z &&
       sd[splatId].fields.position[0] < bbox[1].x && sd[splatId].fields.position[1] < bbox[1].y && sd[splatId].fields.position[2] < bbox[1].z)
       return true;

    return false;
}

class GaussianOctree
{
public:
    GaussianOctree* children[8] = {nullptr};
    std::vector<uint32_t> containedSplats;
    uint8_t level = 0;
    bool isLeaf = false;
    glm::vec3 bbox[2];

    uint32_t representative = 0; /* Will be the splat that is the approximation of all splats contained, will be dynamically added to the array I guess */


    void processSplats(uint8_t _level, std::vector<SplatData> & sd); 
    GaussianOctree( glm::vec3 * _bbox);
    ~GaussianOctree();
};

GaussianOctree::GaussianOctree(glm::vec3 * _bbox)
{
    bbox[0] = _bbox[0];
    bbox[1] = _bbox[1];
}

void computeNodeRepresentative(GaussianOctree * node, std::vector<SplatData>& sd){
    
    if(node->isLeaf){
        float opacityWeight = 0.0f;
        float volumeWeight = 0.0f;

        if(node->containedSplats.size() == 0)
            return;

        SplatData representative;
        for(int i=0;i<62;i++){
            representative.rawData[i] = 0.0f;
        }
        for(auto splat : node->containedSplats){
            float opacity = sd[splat].fields.opacity;
            float volume = sd[splat].fields.scale[0] * sd[splat].fields.scale[1] * sd[splat].fields.scale[2];
            opacityWeight += opacity;
            volumeWeight += volume;

            /* Scaling */
            for(int i = 0; i < 3; i++){
                representative.fields.scale[i] += sd[splat].fields.scale[i] * opacity;
            }

            /* Colors (a.k.a. Sphere harmonics) */
            for(int i = 0; i < 48; i++){
                representative.fields.SH[i] += sd[splat].fields.SH[i] * opacity;
            }

            /* Quaternion */
            for(int i = 0; i < 4; i++){
                representative.fields.rotation[i] += sd[splat].fields.rotation[i] * volume;
            }

            /* Opacity */
            representative.fields.opacity += sd[splat].fields.opacity * volume;

            /* Mean */
            for(int i = 0; i < 3; i++){
                representative.fields.position[i] += sd[splat].fields.position[i] * opacity;
            }

        }

        /* Scaling */
        for(int i = 0; i < 3; i++){
            representative.fields.scale[i] /= opacityWeight;
        }

        /* Colors (a.k.a. Sphere harmonics) */
        for(int i = 0; i < 48; i++){
            representative.fields.SH[i] /= opacityWeight;
        }

        /* Quaternion */
        for(int i = 0; i < 4; i++){
            representative.fields.rotation[i] /= volumeWeight;
        }

        /* Opacity */
        representative.fields.opacity /= volumeWeight;

        /* Mean */
        for(int i = 0; i < 3; i++){
            representative.fields.position[i] /= opacityWeight;
        }
        
        sd.push_back(representative);
        node->representative = sd.size() - 1;
    }
    else{
        int averageNo = 0;

        SplatData representative;
        for(int i=0;i<62;i++){
            representative.rawData[i] = 0.0f;
        }

        for(int k=0;k<8;k++){
            if(node->children[k]->representative!=0){
                averageNo++;
                for(int i=0;i<62;i++){
                    representative.rawData[i] += sd[node->children[k]->representative].rawData[i];
                }
            }
        }

        for(int i=0;i<62;i++){
            representative.rawData[i] /= averageNo;
        }

        sd.push_back(representative);
        node->representative = sd.size() - 1;
    }
}

void GaussianOctree::processSplats(uint8_t _level, std::vector<SplatData> & sd){
    level = _level;

    if(containedSplats.size() == 0){
        isLeaf = true;
        return;
    }

    if(containedSplats.size() < 3){ // Some threshold where it's not worth going deeper
        isLeaf = true;
        computeNodeRepresentative(this, sd);
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
        for(int k = 0; k < containedSplats.size(); k++){
            auto splat = containedSplats[k];
            if(insideBBox(children[i]->bbox, splat, sd)){
                children[i]->containedSplats.push_back(splat);
            }
        }
        if(level < MAX_OCTREE_LEVEL){
            children[i]->isLeaf = false;
            children[i]->processSplats(level+1, sd);
        }
        else{
            children[i]->isLeaf=true;
            computeNodeRepresentative(children[i], sd);
        }

        if(level == 0){
            printf("%i / %i\n", i+1, 8);
        }
    }

    containedSplats.clear();

    computeNodeRepresentative(this, sd);
}

GaussianOctree::~GaussianOctree()
{
    for(int i=0;i<8;i++)
        if(children[i] != nullptr){
            delete children[i];
        }
}

GaussianOctree * buildOctree(std::vector<SplatData> & sd, uint32_t num_primitives){
    glm::vec3 minBound(1e13, 1e13, 1e13);
    glm::vec3 maxBound(-1e13, -1e13, -1e13);

    for(int i = 0; i < num_primitives; i++){
        float maxRadius = max(sd[i].fields.scale[0], max(sd[i].fields.scale[1], sd[i].fields.scale[2]));
        glm::vec3 splatMinBound = glm::make_vec3(sd[i].fields.position) - maxRadius;
        glm::vec3 splatMaxBound = glm::make_vec3(sd[i].fields.position) + maxRadius;
        minBound.x = min(minBound.x, splatMinBound.x);
        minBound.y = min(minBound.y, splatMinBound.y);
        minBound.z = min(minBound.z, splatMinBound.z);
        maxBound.x = max(maxBound.x, splatMaxBound.x);
        maxBound.y = max(maxBound.y, splatMaxBound.y);
        maxBound.z = max(maxBound.z, splatMaxBound.z);
    }

    glm::vec3 center = (minBound + maxBound) * 0.5f;

    float maxSpan = max(maxBound.x - minBound.x, max(maxBound.y - minBound.y, maxBound.z - minBound.z));

    glm::vec3 rootBbox[2];
    rootBbox[0] = center - maxSpan;
    rootBbox[1] = center + maxSpan;

    GaussianOctree * root = new GaussianOctree(rootBbox);
    for(int i = 0; i < num_primitives; i++)
        root->containedSplats.push_back(i);

    root->processSplats(0, sd);

    return root;

}

void markForRender(bool * renderMask, uint32_t num_primitives, GaussianOctree * root, std::vector<SplatData> & sd, int renderLevel = 11){
    if(root->level == renderLevel)
        renderMask[root->representative] = true;
    if(!root->isLeaf){
        for(int i=0;i<8;i++){
            markForRender(renderMask, num_primitives, root->children[i], sd, renderLevel);
        }
    }

}

#endif