#ifndef __GAUSSIAN_OCTREE__
#define __GAUSSIAN_OCTREE__

#define MAX_OCTREE_LEVEL 14

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

    size_t num_fields = sizeof(SplatData) / sizeof(float);
    float nodeSize = (node->bbox[1].x - node->bbox[0].x) / (node->level * node->level);

    /* Accumulation for weighted average */
    float opacityWeight = 0.0f;
    float volumeWeight = 0.0f;

    /* No splats inside, so no representative */
    if(node->containedSplats.size() == 0)
        return;

    /* Only one splat contained */
    if(node->containedSplats.size() == 1){
        node->representative = node->containedSplats[0];
        return;
    }

    /* Representative splat object with empy data */
    SplatData representative;
    for(int i=0;i<num_fields;i++){
        representative.rawData[i] = 0.0f;
    }

    std::vector<glm::vec3> coveragePoints;
    coveragePoints.reserve(node->containedSplats.size() * 7);

    /* Iterate through all the contained splats in the node */
    for(auto splat : node->containedSplats){

        if(sd[splat].fields.opacity < 0.95f)
            continue;

        glm::vec3 e1 = glm::make_vec3(&sd[splat].fields.directions[0]) / 3.0f;
        glm::vec3 e2 = glm::make_vec3(&sd[splat].fields.directions[3]) / 3.0f;
        glm::vec3 e3 = glm::make_vec3(&sd[splat].fields.directions[6]) / 3.0f;

        float opacity = sd[splat].fields.opacity;
        float volume = e1.length() * e2.length() * e3.length();

        opacityWeight += opacity;
        volumeWeight += volume;

        /* Colors (a.k.a. Sphere harmonics) */
        for(int i = 0; i < 48; i++){
            representative.fields.SH[i] += sd[splat].fields.SH[i] * opacity;
        }

        /* Opacity */
        representative.fields.opacity += sd[splat].fields.opacity * volume;

        coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position));
        coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) + e1);
        coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) - e1);
        coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) + e2);
        coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) - e2);
        coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) + e3);
        coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) - e3);

    }

    Eigen::MatrixXf coverageCloud(coveragePoints.size(), 3);
    for(int i = 0; i < coveragePoints.size(); i++){
        coverageCloud(i, 0) = coveragePoints[i].x;
        coverageCloud(i, 1) = coveragePoints[i].y;
        coverageCloud(i, 2) = coveragePoints[i].z;
    }

    // First, we need to compute the mean of the points
    Eigen::Vector3f mean = coverageCloud.colwise().mean();

    // Then, we subtract the mean from the points
    coverageCloud = coverageCloud.rowwise() - mean.transpose();

    // Compute the covariance matrix
    Eigen::Matrix3f cov = coverageCloud.transpose() * coverageCloud;

    // Perform the singular value decomposition
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // The columns of U are the eigenvectors
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Vector3f svals = svd.singularValues();

    representative.fields.covariance[0] = cov(0, 0);
    representative.fields.covariance[1] = cov(0, 1);
    representative.fields.covariance[2] = cov(0, 2);
    representative.fields.covariance[3] = cov(1, 1);
    representative.fields.covariance[4] = cov(1, 2);
    representative.fields.covariance[5] = cov(2, 2);

    for(int i = 0; i < 9; i++){
        representative.fields.directions[i] = U(i % 3, i / 3) * svals(i / 3);
    }

    representative.fields.position[0] = mean(0);
    representative.fields.position[1] = mean(1);
    representative.fields.position[2] = mean(2);

    /* Colors (a.k.a. Sphere harmonics) */
    for(int i = 0; i < 48; i++){
        representative.fields.SH[i] /= opacityWeight;
    }

    /* Opacity */
    representative.fields.opacity /= volumeWeight;
    
    sd.push_back(representative);
    node->representative = sd.size() - 1;
    
}

void GaussianOctree::processSplats(uint8_t _level, std::vector<SplatData> & sd){
    level = _level;

    if(containedSplats.size() == 0){
        isLeaf = true;
        return;
    }

    // if(containedSplats.size() < 3){ // Some threshold where it's not worth going deeper
    //     isLeaf = true;
    //     computeNodeRepresentative(this, sd);
    //     return;
    // }

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

    /* Compute representatives before clearing the contained splats vector */
    computeNodeRepresentative(this, sd);

    containedSplats.clear();

    
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
        minBound.x = min(minBound.x, sd[i].fields.position[0]);
        minBound.y = min(minBound.y, sd[i].fields.position[1]);
        minBound.z = min(minBound.z, sd[i].fields.position[2]);
        maxBound.x = max(maxBound.x, sd[i].fields.position[0]);
        maxBound.y = max(maxBound.y, sd[i].fields.position[1]);
        maxBound.z = max(maxBound.z, sd[i].fields.position[2]);
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

int markForRender(bool * renderMask, uint32_t num_primitives, GaussianOctree * root, std::vector<SplatData> & sd, int renderLevel = 11){
    if(root->containedSplats.size() == 0 && root->representative == 0){
        return 0;
    }
    if(root->level == renderLevel){
        renderMask[root->representative] = true;
        return 1;
    }
    if(root->level <= renderLevel && root->isLeaf){
        for(auto splat : root->containedSplats)
            renderMask[splat] = true;
        return root->containedSplats.size();
    }
    if(!root->isLeaf && root->level < renderLevel){
        int splatsRendered = 0;
        for(int i=0;i<8;i++){
            splatsRendered += markForRender(renderMask, num_primitives, root->children[i], sd, renderLevel);
        }
        return splatsRendered;
    }
    return 0;

}

#endif