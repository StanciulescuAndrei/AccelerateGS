#ifndef __GAUSSIAN_OCTREE__
#define __GAUSSIAN_OCTREE__

#define MAX_OCTREE_LEVEL 18
#define MIN_RESOLUTION 14

#pragma once
#include "PLYReader.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <iostream>

#include <nanoflann.hpp>

#include "GUIManager.h"

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

void addSplatToCoverage(glm::vec3 * coverage, uint32_t splatId, std::vector<SplatData> & sd){
    glm::vec3 splatSpread[2];
    splatSpread[0] = coverage[0];
    splatSpread[1] = coverage[1];

    splatSpread[0] = glm::min(splatSpread[0], glm::make_vec3(sd[splatId].fields.position) - glm::make_vec3(sd[splatId].fields.directions));
    splatSpread[1] = glm::max(splatSpread[1], glm::make_vec3(sd[splatId].fields.position) + glm::make_vec3(sd[splatId].fields.directions));
    splatSpread[0] = glm::min(splatSpread[0], glm::make_vec3(sd[splatId].fields.position) + glm::make_vec3(sd[splatId].fields.directions));
    splatSpread[1] = glm::max(splatSpread[1], glm::make_vec3(sd[splatId].fields.position) - glm::make_vec3(sd[splatId].fields.directions));

    splatSpread[0] = glm::min(splatSpread[0], glm::make_vec3(sd[splatId].fields.position) - glm::make_vec3(sd[splatId].fields.directions + 3));
    splatSpread[1] = glm::max(splatSpread[1], glm::make_vec3(sd[splatId].fields.position) + glm::make_vec3(sd[splatId].fields.directions + 3));
    splatSpread[0] = glm::min(splatSpread[0], glm::make_vec3(sd[splatId].fields.position) + glm::make_vec3(sd[splatId].fields.directions + 3));
    splatSpread[1] = glm::max(splatSpread[1], glm::make_vec3(sd[splatId].fields.position) - glm::make_vec3(sd[splatId].fields.directions + 3));

    splatSpread[0] = glm::min(splatSpread[0], glm::make_vec3(sd[splatId].fields.position) - glm::make_vec3(sd[splatId].fields.directions + 6));
    splatSpread[1] = glm::max(splatSpread[1], glm::make_vec3(sd[splatId].fields.position) + glm::make_vec3(sd[splatId].fields.directions + 6));
    splatSpread[0] = glm::min(splatSpread[0], glm::make_vec3(sd[splatId].fields.position) + glm::make_vec3(sd[splatId].fields.directions + 6));
    splatSpread[1] = glm::max(splatSpread[1], glm::make_vec3(sd[splatId].fields.position) - glm::make_vec3(sd[splatId].fields.directions + 6));

    coverage[0] = glm::min(coverage[0], splatSpread[0]);
    coverage[1] = glm::max(coverage[1], splatSpread[1]);
}

class GaussianOctree
{
public:
    GaussianOctree* children[8] = {nullptr};
    std::vector<uint32_t> containedSplats;
    uint8_t level = 0;
    bool isLeaf = false;
    glm::vec3 bbox[2];
    glm::vec3 coverage[2];

    uint32_t representative = 0; /* Will be the splat that is the approximation of all splats contained, will be dynamically added to the array I guess */


    void processSplats(uint8_t _level, std::vector<SplatData> & sd); 
    GaussianOctree( glm::vec3 * _bbox);
    ~GaussianOctree();
};

GaussianOctree::GaussianOctree(glm::vec3 * _bbox)
{
    bbox[0] = _bbox[0];
    bbox[1] = _bbox[1];

    coverage[0] = _bbox[0];
    coverage[1] = _bbox[1];
}

typedef std::vector<glm::vec3> PointCloud;

struct PointCloudAdaptor {
    const PointCloud &pts;

    PointCloudAdaptor(const PointCloud &pts) : pts(pts) {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the distance between the vector 'p1[0:size-1]' and the data point with index 'idx_p2'
    inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const {
        const float d0 = p1[0] - pts[idx_p2].x;
        const float d1 = p1[1] - pts[idx_p2].y;
        const float d2 = p1[2] - pts[idx_p2].z;
        return d0 * d0 + d1 * d1 + d2 * d2;
    }

    // Returns the dim'th component of the idx'th point in the class
    inline float kdtree_get_pt(const size_t idx, int dim) const {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
    PointCloudAdaptor,
    3 /* dimension */
> KDTree;

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

    PointCloud coveragePoints;
    coveragePoints.reserve(node->containedSplats.size() * 7);

    std::vector<float> opacities;

    /* Iterate through all the contained splats in the node */
    for(auto splat : node->containedSplats){

        if(sd[splat].fields.opacity < 0.01f)
            continue;

        glm::vec3 e1 = glm::make_vec3(&sd[splat].fields.directions[0]) * 1.41f; // * 1.71f
        glm::vec3 e2 = glm::make_vec3(&sd[splat].fields.directions[3]) * 1.41f; // * 1.71f
        glm::vec3 e3 = glm::make_vec3(&sd[splat].fields.directions[6]) * 1.41f; // * 1.71f

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

        for(int k = 0; k < 7; k++){
            opacities.push_back(opacity);
        }
    }

    if(coveragePoints.size() == 0){
        node->representative = 0;
        return;
    }

    /* Compute point densities */
    std::vector<float> densities;
    densities.reserve(coveragePoints.size());
    for(int i = 0; i < coveragePoints.size(); i++){
        densities.push_back(0);
    }

    float min_density = 1e10;
    float max_density = 0.0f;
    omp_set_num_threads(4);

    PointCloudAdaptor pcAdaptor(coveragePoints);

    // Build the KD-tree index
    KDTree index(3, pcAdaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    // Number of neighbors to consider
    size_t k = 10;

    for (size_t i = 0; i < coveragePoints.size(); ++i) {
        k = std::min((int)coveragePoints.size(), 10);
        std::vector<uint32_t> indices(k);
        std::vector<float> dists(k);
        index.knnSearch(&coveragePoints[i][0], k, &indices[0], &dists[0]);

        // Compute density as inverse of the mean distance
        densities[i] = k / std::accumulate(dists.begin(), dists.end(), 0.0);
        if(densities[i] < min_density)
            min_density = densities[i];
        if(densities[i] > max_density)
            max_density = densities[i];
        
    }

    for(int i = 0; i < coveragePoints.size(); i++){
        densities[i] = 1.0f; // - (densities[i] - min_density) / (max_density - min_density) * 0.5; //0.5f + opacities[i] * 0.5f 
        densities[i] = std::max(densities[i], 0.0f);
    }

    float sum_density = std::accumulate(densities.begin(), densities.end(), 0.0);

    for(int i = 0; i < coveragePoints.size(); i++){
        densities[i] = densities[i] / sum_density;
    }

    Eigen::MatrixXf coverageCloud(coveragePoints.size(), 3);
    for(int i = 0; i < coveragePoints.size(); i++){
        coverageCloud(i, 0) = coveragePoints[i].x;
        coverageCloud(i, 1) = coveragePoints[i].y;
        coverageCloud(i, 2) = coveragePoints[i].z;
    }

    // First, we need to compute the mean of the points
    Eigen::Vector3f weighted_mean;
    weighted_mean << 0.0f, 0.0f, 0.0f;
    for(int i = 0; i < coveragePoints.size(); i++){
        weighted_mean(0) += (coverageCloud(i, 0) * densities[i]);
        weighted_mean(1) += (coverageCloud(i, 1) * densities[i]);
        weighted_mean(2) += (coverageCloud(i, 2) * densities[i]);
    }

    Eigen::Vector3f mean = coverageCloud.colwise().mean();

    // Then, we subtract the mean from the points
    coverageCloud = coverageCloud.rowwise() - weighted_mean.transpose();

    const int n = coveragePoints.size();
    Eigen::DiagonalMatrix<float, Eigen::Dynamic> W(n);
    for(int i=0;i<n;i++){
        W.diagonal()[i] = densities[i];
    }

    // Compute the covariance matrix
    Eigen::Matrix3f cov = coverageCloud.transpose() * W * coverageCloud;

    // cov = cov / (weightsum * weightsum); /* NOT the proper formula: https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis */

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
        representative.fields.directions[i] = U(i % 3, i / 3)* svals(i / 3);
    }

    representative.fields.position[0] = weighted_mean(0);
    representative.fields.position[1] = weighted_mean(1);
    representative.fields.position[2] = weighted_mean(2);

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
        representative = 0;
        return;
    }

    if(containedSplats.size() == 1){
        isLeaf = true;
        representative = containedSplats[0];
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
                addSplatToCoverage(children[i]->coverage, splat, sd);
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
    if(this->level >= MIN_RESOLUTION)
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

int markForRender(bool * renderMask, uint32_t num_primitives, GaussianOctree * root, std::vector<SplatData> & sd, int renderLevel, glm::vec3 & cameraPosition, float fovy, int SW, float dpt){

    if(renderLevel == -1){
        int shouldRenderNode = 0;
        if(root == nullptr)
            return 0;
        /* Easiest implementation, maximum projection by distance */
        float S = glm::length(root->coverage[0] - root->coverage[1]);
        float D = glm::length((root->coverage[0] + root->coverage[1]) / 2.0f - cameraPosition);

        float P = S / D * (SW / fovy);

        shouldRenderNode = (P > dpt);

        if(shouldRenderNode){ // is node big enough on the screen?
            if(root->isLeaf && root->containedSplats.size() > 0){
                for(auto splat : root->containedSplats)
                    renderMask[splat] = true;
                return root->containedSplats.size();
            }
            else{
                int splatsRendered = 0;
                for(int i=0;i<8;i++){
                    splatsRendered += markForRender(renderMask, num_primitives, root->children[i], sd, renderLevel, cameraPosition, fovy, SW, dpt);
                }
                return splatsRendered;
            }
        }
        else{
            if(root->representative != 0){
                renderMask[root->representative] = true;
                return 1;
            }
            else{ // Level too low to have a representative, still have to go down
                int splatsRendered = 0;
                for(int i=0;i<8;i++){
                    if(root->children[i] != nullptr)
                        splatsRendered += markForRender(renderMask, num_primitives, root->children[i], sd, renderLevel, cameraPosition, fovy, SW, dpt);
                }
                return splatsRendered;
            }
        }
    }
    else{
        if(root->level == renderLevel){
            renderMask[root->representative] = true;
            return 1;
        }
        if(root->level < renderLevel && root->isLeaf){
            for(auto splat : root->containedSplats)
                renderMask[splat] = true;
            return root->containedSplats.size();
        }
        if(!root->isLeaf && root->level < renderLevel){
            int splatsRendered = 0;
            for(int i=0;i<8;i++){
                splatsRendered += markForRender(renderMask, num_primitives, root->children[i], sd, renderLevel, cameraPosition, fovy, SW, dpt);
            }
            return splatsRendered;
        }
        if(root->containedSplats.size() == 0 && root->representative == 0){
            return 0;
        }
    }
    
    
    
    return 0;

}

#endif