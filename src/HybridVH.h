#ifndef __HYBRID_VH__
#define __HYBRID_VH__

#pragma once
#include "PLYReader.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <iostream>

#include <nanoflann.hpp>

#include "GUIManager.h"
#include "GaussianOctree.h"

#include "PointClustering.h"

enum LevelType
{
    OctreeLevel,
    BipartitionLevel
};

struct CUDATreeNode{
    uint32_t childrenIndices[2];
    uint32_t splatIds[4]; // For alignement, whatever
    uint8_t flags;
    uint8_t level;
    float3 center;
    float diagonal;
    uint32_t representative;
};

class HybridVH : public SpacePartitioningBase
{
public:
    std::vector<HybridVH *> children;
    std::vector<uint32_t> * containedSplats;
    LevelType levelType;
    uint8_t level = 0;
    bool isLeaf = false;
    glm::vec3 bbox[2];
    glm::vec3 coverage[2];

    uint32_t representative = 0; /* Will be the splat that is the approximation of all splats contained, will be dynamically added to the array I guess */

    void processSplats(uint8_t _level, std::vector<SplatData> &sd, volatile int *progress);
    void buildVHStructure(std::vector<SplatData> &sd, uint32_t num_primitives, volatile int *progress) override;
    int markForRender(bool *renderMask, uint32_t num_primitives, int renderLevel, glm::vec3 &cameraPosition, float fovy, int SW, float dpt) override;
    HybridVH(glm::vec3 *_bbox);
    HybridVH();
    ~HybridVH();
};

HybridVH::HybridVH(glm::vec3 *_bbox)
{
    bbox[0] = _bbox[0];
    bbox[1] = _bbox[1];

    coverage[0] = _bbox[0];
    coverage[1] = _bbox[1];

    containedSplats = new std::vector<uint32_t>();
}

HybridVH::HybridVH()
{
    glm::vec3 zero(0.0f);
    bbox[0] = zero;
    bbox[1] = zero;

    coverage[0] = zero;
    coverage[1] = zero;

    containedSplats = new std::vector<uint32_t>();
}

// void computeNodeRepresentative(HybridVH *node, std::vector<SplatData> &sd)
// {

//     if (node == nullptr || node->containedSplats->size() == 0)
//         return;

//     if (renderConfig.representative == std::string("pca"))
//     {
//         size_t num_fields = sizeof(SplatData) / sizeof(float);
//         float nodeSize = (node->bbox[1].x - node->bbox[0].x) / (node->level * node->level);

//         /* Accumulation for weighted average */
//         float splatWeight = 0.0f;

//         /* Representative splat object with empy data */
//         SplatData representative;
//         for (int i = 0; i < num_fields; i++)
//         {
//             representative.rawData[i] = 0.0f;
//         }

//         PointCloud coveragePoints;
//         coveragePoints.reserve(node->containedSplats->size() * 7);

//         std::vector<float> pointWeights;

//         std::vector<uint32_t> base_splats;
//         base_splats = *(node->containedSplats);

//         // for (auto child : node->children)
//         // {
//         //     if (child != nullptr && child->representative != 0)
//         //     {
//         //         base_splats.push_back(child->representative);
//         //     }
//         // }
//         // if (base_splats.size() == 0)
//         // {
//         //     base_splats = node->containedSplats;
//         // }

//         // if (base_splats.size() == 0)
//         // {
//         //     node->representative = 0;
//         //     return;
//         // }

//         base_splats.erase(std::remove_if(
//                               base_splats.begin(),
//                               base_splats.end(),
//                               [&](uint32_t k)
//                               {
//                                   if (k < sd.size())
//                                       return (sd[k].fields.opacity < OPACITY_THRESHOLD);
//                                   return true;
//                               }),
//                           base_splats.end());

//         if (base_splats.size() == 0)
//         {
//             node->representative = 0;
//             return;
//         }

//         if (base_splats.size() == 1)
//         {
//             node->representative = base_splats[0]; // base_splats[0]
//             return;
//         }

//         /* Iterate through all the contained splats in the node */
//         for (auto splat : base_splats)
//         {

//             glm::vec3 e1 = glm::make_vec3(&sd[splat].fields.directions[0]) * 3.0f;
//             glm::vec3 e2 = glm::make_vec3(&sd[splat].fields.directions[3]) * 3.0f;
//             glm::vec3 e3 = glm::make_vec3(&sd[splat].fields.directions[6]) * 3.0f;

//             float opacity = sd[splat].fields.opacity;
//             float volume = e1.length() * e2.length() * e3.length();

//             float individualSplatWeight = opacity * volume; // glm::pow(volume, 0.33)
//             splatWeight += individualSplatWeight;

//             /* Colors (a.k.a. Sphere harmonics) */
//             for (int i = 0; i < 48; i++)
//             {
//                 representative.fields.SH[i] += sd[splat].fields.SH[i] * individualSplatWeight;
//             }

//             coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position));
//             coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) + e1);
//             coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) - e1);
//             coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) + e2);
//             coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) - e2);
//             coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) + e3);
//             coveragePoints.push_back(glm::make_vec3(sd[splat].fields.position) - e3);

//             for (int k = 0; k < 7; k++)
//             {
//                 pointWeights.push_back(individualSplatWeight);
//             }
//         }

//         /* Compute point densities */
//         std::vector<float> densities;
//         densities.reserve(coveragePoints.size());
//         for (int i = 0; i < coveragePoints.size(); i++)
//         {
//             densities.push_back(0);
//         }

//         float min_density = 1e10;
//         float max_density = 0.0f;

//         PointCloudAdaptor pcAdaptor(coveragePoints);

//         // Build the KD-tree index
//         KDTree index(3, pcAdaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
//         index.buildIndex();

//         // Number of neighbors to consider
//         size_t k = 10;

//         for (size_t i = 0; i < coveragePoints.size(); ++i)
//         {
//             k = std::min((int)coveragePoints.size(), 10);
//             std::vector<uint32_t> indices(k);
//             std::vector<float> dists(k);
//             index.knnSearch(&coveragePoints[i][0], k, &indices[0], &dists[0]);

//             // Compute density as inverse of the mean distance
//             densities[i] = k / std::accumulate(dists.begin(), dists.end(), 0.0);
//             if (densities[i] < min_density)
//                 min_density = densities[i];
//             if (densities[i] > max_density)
//                 max_density = densities[i];
//         }

//         for (int i = 0; i < coveragePoints.size(); i++)
//         {
//             densities[i] = pointWeights[i]; // pointWeights[i]; /* Important stuff here */
//             densities[i] = std::max(densities[i], 0.001f);
//         }

//         float sum_density = std::accumulate(densities.begin(), densities.end(), 0.0);

//         for (int i = 0; i < coveragePoints.size(); i++)
//         {
//             densities[i] = densities[i] / sum_density;
//         }

//         Eigen::MatrixXf coverageCloud(coveragePoints.size(), 3);
//         for (int i = 0; i < coveragePoints.size(); i++)
//         {
//             coverageCloud(i, 0) = coveragePoints[i].x;
//             coverageCloud(i, 1) = coveragePoints[i].y;
//             coverageCloud(i, 2) = coveragePoints[i].z;
//         }

//         // First, we need to compute the mean of the points
//         Eigen::Vector3f weighted_mean;
//         weighted_mean << 0.0f, 0.0f, 0.0f;
//         for (int i = 0; i < coveragePoints.size(); i++)
//         {
//             weighted_mean(0) += (coverageCloud(i, 0) * densities[i]);
//             weighted_mean(1) += (coverageCloud(i, 1) * densities[i]);
//             weighted_mean(2) += (coverageCloud(i, 2) * densities[i]);
//         }

//         Eigen::Vector3f mean = coverageCloud.colwise().mean();

//         // Then, we subtract the mean from the points
//         coverageCloud = coverageCloud.rowwise() - weighted_mean.transpose();

//         const int n = coveragePoints.size();
//         Eigen::DiagonalMatrix<float, Eigen::Dynamic> W(n);
//         for (int i = 0; i < n; i++)
//         {
//             W.diagonal()[i] = densities[i];
//         }

//         // Compute the covariance matrix
//         Eigen::Matrix3f cov = coverageCloud.transpose() * W * coverageCloud;

//         // cov = cov / (weightsum * weightsum); /* NOT the proper formula: https://stats.stackexchange.com/questions/113485/weighted-principal-components-analysis */

//         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov);

//         if (eigensolver.info() != Eigen::Success)
//         {
//             printf("Error in eigen-decomposition!!!\n");
//         }

//         Eigen::Vector3f eigenvalues = eigensolver.eigenvalues();
//         Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors();

//         /* Deal with negative eigenvalues */
//         /* No need to flip vectors as they define the ellipsoid so the direction is irrelevant */
//         eigenvalues = eigenvalues.cwiseAbs();

//         Eigen::Vector3f axes_lengths = eigenvalues.cwiseSqrt();

//         axes_lengths *= 2.4477;

//         representative.fields.covariance[0] = cov(0, 0);
//         representative.fields.covariance[1] = cov(0, 1);
//         representative.fields.covariance[2] = cov(0, 2);
//         representative.fields.covariance[3] = cov(1, 1);
//         representative.fields.covariance[4] = cov(1, 2);
//         representative.fields.covariance[5] = cov(2, 2);

//         for (int i = 0; i < 9; i++)
//         {
//             representative.fields.directions[i] = eigenvectors.col(i / 3)(i % 3) / 10.0f * axes_lengths[i / 3]; // * axes_lengths[i / 3]
//             if (isnanf(representative.fields.directions[i]))
//             {
//                 printf("broke here idk\n");
//             }
//         }
//         // representative.fields.directions[0] = 0.01f;
//         // representative.fields.directions[1] = 0.0f;
//         // representative.fields.directions[2] = 0.0f;
//         // representative.fields.directions[3] = 0.0f;
//         // representative.fields.directions[4] = 0.01f;
//         // representative.fields.directions[5] = 0.0f;
//         // representative.fields.directions[6] = 0.0f;
//         // representative.fields.directions[7] = 0.0f;
//         // representative.fields.directions[8] = 0.01f;

//         representative.fields.position[0] = weighted_mean(0);
//         representative.fields.position[1] = weighted_mean(1);
//         representative.fields.position[2] = weighted_mean(2);

//         /* Colors (a.k.a. Sphere harmonics) */
//         for (int i = 0; i < 48; i++)
//         {
//             representative.fields.SH[i] /= splatWeight;
//         }

//         /* Opacity */

//         float approx_splat_volume = std::max(glm::abs(cov.determinant()), 0.001f);
//         representative.fields.opacity = 0.0f;

//         for (auto w : densities)
//         {
//             representative.fields.opacity += (w / (glm::pow(approx_splat_volume, 0.33) * 9));
//         }

//         sd.push_back(representative);
//         node->representative = sd.size() - 1;
//     }
//     else if (renderConfig.representative == std::string("inria"))
//     {
//         size_t num_fields = sizeof(SplatData) / sizeof(float);
//         float nodeSize = (node->bbox[1].x - node->bbox[0].x) / (node->level * node->level);

//         std::vector<uint32_t> base_splats;
//         if (node->isLeaf)
//         {
//             base_splats = *(node->containedSplats);
//         }
//         else
//         {
//             for (HybridVH * child : node->children)
//             {
//                 if (child != nullptr && child->representative != 0)
//                 {
//                     base_splats.push_back(child->representative);
//                 }
//             }
//         }

//         /* Representative splat object with empy data */
//         SplatData representative;
//         for (int i = 0; i < num_fields; i++)
//         {
//             representative.rawData[i] = 0.0f;
//         }
//         std::vector<float> weights;

//         /* Iterate through all the contained splats in the node */
//         bool worthit = false;
//         for (auto splat : base_splats)
//         {

//             glm::vec3 e1 = glm::make_vec3(&sd[splat].fields.directions[0]) * 3.0f;
//             glm::vec3 e2 = glm::make_vec3(&sd[splat].fields.directions[3]) * 3.0f;
//             glm::vec3 e3 = glm::make_vec3(&sd[splat].fields.directions[6]) * 3.0f;

//             float opacity = sd[splat].fields.opacity;
//             float volume = e1.length() * e2.length() * e3.length();

//             if (sd[splat].fields.opacity < OPACITY_THRESHOLD)
//             {
//                 weights.push_back(0.0f);
//             }
//             else
//             {
//                 weights.push_back(opacity * glm::pow(volume, 0.33));
//                 worthit = true;
//             }

//             /* Opacity */
//             representative.fields.opacity += sd[splat].fields.opacity * volume;
//         }
//         if (!worthit)
//         {
//             node->representative = base_splats[0];
//             return;
//         }

//         float sum_weight = std::accumulate(weights.begin(), weights.end(), 0.0001f);

//         glm::vec3 weighted_mean(0.0f);
//         Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();

//         int idx = 0;
//         // First, do the SHs and the weighted mean (a.k.a position)
//         for (auto splat : base_splats)
//         {
//             weights[idx] /= sum_weight;

//             /* Colors (a.k.a. Sphere harmonics) */
//             for (int i = 0; i < 48; i++)
//             {
//                 representative.fields.SH[i] += sd[splat].fields.SH[i] * weights[idx];
//             }

//             weighted_mean += glm::make_vec3(sd[splat].fields.position) * weights[idx];

//             idx++;
//         }

//         Eigen::Vector3f w_mean;
//         w_mean << weighted_mean.x, weighted_mean.y, weighted_mean.z;

//         idx = 0;
//         // second, do the covariance merging
//         for (auto splat : base_splats)
//         {

//             Eigen::Matrix3f crt_cov;
//             crt_cov << sd[splat].fields.covariance[0], sd[splat].fields.covariance[1], sd[splat].fields.covariance[2],
//                 sd[splat].fields.covariance[1], sd[splat].fields.covariance[3], sd[splat].fields.covariance[4],
//                 sd[splat].fields.covariance[2], sd[splat].fields.covariance[4], sd[splat].fields.covariance[5];

//             Eigen::Vector3f mean;
//             mean << sd[splat].fields.position[0], sd[splat].fields.position[1], sd[splat].fields.position[2];

//             cov += weights[idx] * (crt_cov + (mean - w_mean) * (mean - w_mean).transpose());

//             idx++;
//         }

//         /* Opacity */
//         float approx_splat_volume = std::max(glm::abs(cov.determinant()), 0.001f);
//         representative.fields.opacity = 0.0f;

//         for (auto w : weights)
//         {
//             representative.fields.opacity += (w / (glm::pow(approx_splat_volume, 0.33) * 9)); /// (glm::pow(approx_splat_volume, 0.33) * 9)
//         }

//         representative.fields.covariance[0] = cov(0, 0);
//         representative.fields.covariance[1] = cov(0, 1);
//         representative.fields.covariance[2] = cov(0, 2);
//         representative.fields.covariance[3] = cov(1, 1);
//         representative.fields.covariance[4] = cov(1, 2);
//         representative.fields.covariance[5] = cov(2, 2);

//         representative.fields.position[0] = weighted_mean.x;
//         representative.fields.position[1] = weighted_mean.y;
//         representative.fields.position[2] = weighted_mean.z;

//         sd.push_back(representative);
//         node->representative = sd.size() - 1;
//     }
// }

void computeNodeRepresentative(HybridVH *node, std::vector<SplatData> &sd)
{

    if (renderConfig.representative == std::string("pca"))
    {
        size_t num_fields = sizeof(SplatData) / sizeof(float);
        float nodeSize = (node->bbox[1].x - node->bbox[0].x) / (node->level * node->level);

        /* Accumulation for weighted average */
        float opacityWeight = 0.0f;
        float volumeWeight = 0.0f;

        /* No splats inside, so no representative */
        if (node->containedSplats->size() == 0)
            return;

        /* Only one splat contained */
        if (node->containedSplats->size() == 1)
        {
            node->representative = node->containedSplats->at(0);
            return;
        }

        /* Representative splat object with empy data */
        SplatData representative;
        for (int i = 0; i < num_fields; i++)
        {
            representative.rawData[i] = 0.0f;
        }

        PointCloud coveragePoints;
        coveragePoints.reserve(node->containedSplats->size() * 7);

        std::vector<float> opacities;

        /* Iterate through all the contained splats in the node */
        for (auto splat : *(node->containedSplats))
        {

            if (sd[splat].fields.opacity < OPACITY_THRESHOLD)
                continue;

            glm::vec3 e1 = glm::make_vec3(&sd[splat].fields.directions[0]) * 3.0f;
            glm::vec3 e2 = glm::make_vec3(&sd[splat].fields.directions[3]) * 3.0f;
            glm::vec3 e3 = glm::make_vec3(&sd[splat].fields.directions[6]) * 3.0f;

            float opacity = sd[splat].fields.opacity;
            float volume = e1.length() * e2.length() * e3.length();

            opacityWeight += opacity;
            volumeWeight += volume;

            /* Colors (a.k.a. Sphere harmonics) */
            for (int i = 0; i < 48; i++)
            {
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

            for (int k = 0; k < 7; k++)
            {
                opacities.push_back(opacity);
            }
        }

        if (coveragePoints.size() == 0)
        {
            node->representative = 0;
            return;
        }

        /* Compute point densities */
        std::vector<float> densities;
        densities.reserve(coveragePoints.size());
        for (int i = 0; i < coveragePoints.size(); i++)
        {
            densities.push_back(0);
        }

        float min_density = 1e10;
        float max_density = 0.0f;

        PointCloudAdaptor pcAdaptor(coveragePoints);

        // Build the KD-tree index
        KDTree index(3, pcAdaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        // Number of neighbors to consider
        size_t k = 10;

        for (size_t i = 0; i < coveragePoints.size(); ++i)
        {
            k = std::min((int)coveragePoints.size(), 10);
            std::vector<uint32_t> indices(k);
            std::vector<float> dists(k);
            index.knnSearch(&coveragePoints[i][0], k, &indices[0], &dists[0]);

            // Compute density as inverse of the mean distance
            densities[i] = k / std::accumulate(dists.begin(), dists.end(), 0.0);
            if (densities[i] < min_density)
                min_density = densities[i];
            if (densities[i] > max_density)
                max_density = densities[i];
        }

        for (int i = 0; i < coveragePoints.size(); i++)
        {
            // densities[i] = opacities[i] - (densities[i] - min_density) / (max_density - min_density) * 0.5;
            // densities[i] = opacities[i];
            densities[i] = 1.0f - (densities[i] - min_density) / (max_density - min_density) * 0.5;
            // densities[i] = 1.0f;
            densities[i] = std::max(densities[i], 0.001f);
        }

        float sum_density = std::accumulate(densities.begin(), densities.end(), 0.0);

        for (int i = 0; i < coveragePoints.size(); i++)
        {
            densities[i] = densities[i] / sum_density;
        }

        Eigen::MatrixXf coverageCloud(coveragePoints.size(), 3);
        for (int i = 0; i < coveragePoints.size(); i++)
        {
            coverageCloud(i, 0) = coveragePoints[i].x;
            coverageCloud(i, 1) = coveragePoints[i].y;
            coverageCloud(i, 2) = coveragePoints[i].z;
        }

        // First, we need to compute the mean of the points
        Eigen::Vector3f weighted_mean;
        weighted_mean << 0.0f, 0.0f, 0.0f;
        for (int i = 0; i < coveragePoints.size(); i++)
        {
            weighted_mean(0) += (coverageCloud(i, 0) * densities[i]);
            weighted_mean(1) += (coverageCloud(i, 1) * densities[i]);
            weighted_mean(2) += (coverageCloud(i, 2) * densities[i]);
        }

        Eigen::Vector3f mean = coverageCloud.colwise().mean();

        // Then, we subtract the mean from the points
        coverageCloud = coverageCloud.rowwise() - weighted_mean.transpose();

        const int n = coveragePoints.size();
        Eigen::DiagonalMatrix<float, Eigen::Dynamic> W(n);
        for (int i = 0; i < n; i++)
        {
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
        Eigen::Matrix3f V = svd.matrixV().transpose();

        Eigen::DiagonalMatrix<float, 3> diag(svals(1), svals(0), svals(2));

        // cov = U * diag * V;

        representative.fields.covariance[0] = cov(0, 0);
        representative.fields.covariance[1] = cov(0, 1);
        representative.fields.covariance[2] = cov(0, 2);
        representative.fields.covariance[3] = cov(1, 1);
        representative.fields.covariance[4] = cov(1, 2);
        representative.fields.covariance[5] = cov(2, 2);

        // for(int i = 0; i < 9; i++){
        //     representative.fields.directions[i] = U(i % 3, i / 3) * svals(i / 3);
        // }

        representative.fields.position[0] = weighted_mean(0);
        representative.fields.position[1] = weighted_mean(1);
        representative.fields.position[2] = weighted_mean(2);

        /* Colors (a.k.a. Sphere harmonics) */
        for (int i = 0; i < 48; i++)
        {
            representative.fields.SH[i] /= opacityWeight;
        }

        /* Opacity */
        representative.fields.opacity /= volumeWeight;

        sd.push_back(representative);
        node->representative = sd.size() - 1;
    }
    else if (renderConfig.representative == std::string("inria"))
    {
        size_t num_fields = sizeof(SplatData) / sizeof(float);
        float nodeSize = (node->bbox[1].x - node->bbox[0].x) / (node->level * node->level);

        /* No splats inside, so no representative */
        if (node->containedSplats->size() == 0)
            return;

        /* Only one splat contained */
        if (node->containedSplats->size() == 1)
        {
            node->representative = node->containedSplats->at(0);
            return;
        }

        /* Representative splat object with empy data */
        SplatData representative;
        for (int i = 0; i < num_fields; i++)
        {
            representative.rawData[i] = 0.0f;
        }
        std::vector<float> weights;

        /* Iterate through all the contained splats in the node */
        bool worthit = false;
        for (auto splat : *(node->containedSplats))
        {

            glm::vec3 e1 = glm::make_vec3(&sd[splat].fields.directions[0]) * 3.0f;
            glm::vec3 e2 = glm::make_vec3(&sd[splat].fields.directions[3]) * 3.0f;
            glm::vec3 e3 = glm::make_vec3(&sd[splat].fields.directions[6]) * 3.0f;

            float opacity = sd[splat].fields.opacity;
            float volume = e1.length() * e2.length() * e3.length();

            if (sd[splat].fields.opacity < OPACITY_THRESHOLD)
            {
                weights.push_back(0.0f);
            }
            else
            {
                weights.push_back(opacity * glm::pow(volume, 0.33));
                worthit = true;
            }

            /* Opacity */
            representative.fields.opacity += sd[splat].fields.opacity * volume;
        }
        if (!worthit)
        {
            node->representative = node->containedSplats->at(0);
            return;
        }

        float sum_weight = std::accumulate(weights.begin(), weights.end(), 0.0001f);

        glm::vec3 weighted_mean(0.0f);
        Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();

        int idx = 0;
        // First, do the SHs and the weighted mean (a.k.a position)
        for (auto splat : *(node->containedSplats))
        {
            weights[idx] /= sum_weight;

            /* Colors (a.k.a. Sphere harmonics) */
            for (int i = 0; i < 48; i++)
            {
                representative.fields.SH[i] += sd[splat].fields.SH[i] * weights[idx];
            }

            weighted_mean += glm::make_vec3(sd[splat].fields.position) * weights[idx];

            idx++;
        }

        Eigen::Vector3f w_mean;
        w_mean << weighted_mean.x, weighted_mean.y, weighted_mean.z;

        idx = 0;
        // second, do the covariance merging
        for (auto splat : *(node->containedSplats))
        {

            Eigen::Matrix3f crt_cov;
            crt_cov << sd[splat].fields.covariance[0], sd[splat].fields.covariance[1], sd[splat].fields.covariance[2],
                sd[splat].fields.covariance[1], sd[splat].fields.covariance[3], sd[splat].fields.covariance[4],
                sd[splat].fields.covariance[2], sd[splat].fields.covariance[4], sd[splat].fields.covariance[5];

            Eigen::Vector3f mean;
            mean << sd[splat].fields.position[0], sd[splat].fields.position[1], sd[splat].fields.position[2];

            cov += weights[idx] * (crt_cov + (mean - w_mean) * (mean - w_mean).transpose());

            idx++;
        }

        /* Opacity */
        float approx_splat_volume = std::max(glm::abs(cov.determinant()), 0.001f);
        representative.fields.opacity = 0.0f;

        for (auto w : weights)
        {
            representative.fields.opacity += (w / (glm::pow(approx_splat_volume, 0.33) * 5)); /// (glm::pow(approx_splat_volume, 0.33) * 9)
        }

        representative.fields.covariance[0] = cov(0, 0);
        representative.fields.covariance[1] = cov(0, 1);
        representative.fields.covariance[2] = cov(0, 2);
        representative.fields.covariance[3] = cov(1, 1);
        representative.fields.covariance[4] = cov(1, 2);
        representative.fields.covariance[5] = cov(2, 2);

        representative.fields.position[0] = weighted_mean.x;
        representative.fields.position[1] = weighted_mean.y;
        representative.fields.position[2] = weighted_mean.z;

        sd.push_back(representative);
        node->representative = sd.size() - 1;
    }
}

void HybridVH::processSplats(uint8_t _level, std::vector<SplatData> &sd, volatile int *progress)
{
    level = _level;

    if (level == 3)
    {
        (*progress)++;
    }

    if (containedSplats->size() == 0)
    {
        isLeaf = true;
        representative = 0;
        return;
    }
    // if (containedSplats->size() == 1)
    // {
    //     isLeaf = true;
    //     representative = (*containedSplats)[0];
    //     return;
    // }

    if (level < MIN_HYBRID_LEVEL)
    {
        levelType = OctreeLevel;
        /* Process this as if it were an octree node */
        glm::vec3 halfSize = (bbox[1] - bbox[0]) * 0.5f;

        for (int i = 0; i < 8; i++)
        {
            // Define node's BBox
            glm::vec3 childBbox[2];
            childBbox[0][0] = bbox[0][0] + ((i & 0b001) != 0) * halfSize[0];
            childBbox[0][1] = bbox[0][1] + ((i & 0b010) != 0) * halfSize[1];
            childBbox[0][2] = bbox[0][2] + ((i & 0b100) != 0) * halfSize[2];
            childBbox[1] = childBbox[0] + halfSize;

            HybridVH *child = new HybridVH(childBbox);

            // See which of the splats go into the newly created node
            for (int k = 0; k < containedSplats->size(); k++)
            {
                auto splat = (*containedSplats)[k];
                if (insideBBox(child->bbox, splat, sd))
                {
                    child->containedSplats->push_back(splat);
                    addSplatToCoverage(child->coverage, splat, sd);
                }
            }
            child->isLeaf = false;
            child->representative = 0;
            if(child->containedSplats->size() > 0)
                children.push_back(child);
            else 
                delete child;
        }
        // computeNodeRepresentative(this, sd);
        containedSplats->clear();

        // for(int c = 0; c < children.size(); c++){
        //     if(children[c]->containedSplats->size() == 1){
        //         for(int n = 0; n < children.size(); n++){
        //             if(n == c) continue;
        //             if(children[n]->containedSplats->size() > 0){
        //                 children[n]->isLeaf = false;
        //                 children[n]->containedSplats->push_back((*children[c]->containedSplats)[0]);
        //                 addSplatToCoverage(children[n]->coverage, (*children[c]->containedSplats)[0], sd);
        //                 children[c]->containedSplats->clear();
        //                 children[c]->isLeaf = true;
        //                 children[c]->representative = 0;
        //                 break;
        //             }
        //         }
        //     }
        // }

        for(auto child : children){
            child->processSplats(level + 1, sd, progress);
        }
        
    }
    else
    {
        levelType = BipartitionLevel;
        // containedSplats.erase(std::remove_if(
        //                       containedSplats.begin(),
        //                       containedSplats.end(),
        //                       [&](uint32_t k)
        //                       {
        //                           if (k < sd.size())
        //                               return (sd[k].fields.opacity < OPACITY_THRESHOLD);
        //                           return true;
        //                       }),
        //                   containedSplats.end());
        
        if (containedSplats->size() == 0)
        {
            isLeaf = true;
            representative = 0;
            return;
        }
        if (containedSplats->size() == 1)
        {
            isLeaf = true;
            representative = (*containedSplats)[0];
            return;
        }

        std::vector<int> assignment;
        assignment.reserve(containedSplats->size());

        computeNodeRepresentative(this, sd);

        if(containedSplats->size() > renderConfig.nClusters){
            if(renderConfig.clustering == std::string("pcdecomp"))
                PCReprojectionClustering(coverage, *containedSplats, sd, renderConfig.nClusters, assignment);
            else if(renderConfig.clustering == std::string("spectral"))
                printf("Spectral clustering support dropped!\n"); //SpectralClustering(coverage, *containedSplats, sd, renderConfig.nClusters, assignment);
            for(int i = 0; i < renderConfig.nClusters; i++){
                HybridVH *child = new HybridVH();
                // See which of the splats go into the newly created node
                for (int k = 0; k < containedSplats->size(); k++)
                {
                    if(assignment[k] == i){
                        auto splat = (*containedSplats)[k];
                        child->containedSplats->push_back(splat);
                        addSplatToCoverage(child->coverage, splat, sd);
                    }
                }
                /* Actual split is not that relevant */
                child->bbox[0] = child->coverage[0];
                child->bbox[1] = child->coverage[1];

                children.push_back(child);
            }
        }
        else{
            /* Binary split nodes, here we do da smart stuff */
            glm::vec3 halfSize = (bbox[1] - bbox[0]);

            /* Find the largest dimension of the initial box */
            std::vector<float> projs;
            int maxDim = -1;
            if (halfSize.x > max(halfSize.y, halfSize.z))
            {
                maxDim = 0;
            }
            else if (halfSize.y > max(halfSize.x, halfSize.z))
            {
                maxDim = 1;
            }
            else
            {
                maxDim = 2;
            }

            for (auto splat : *containedSplats)
            {
                projs.push_back(sd[splat].fields.position[maxDim]);
            }

            std::sort(projs.begin(), projs.end());

            float median = (projs[projs.size() / 2] + projs[projs.size() / 2 - 1]) / 2.0f;

            glm::vec3 childBbox1[2];
            childBbox1[0][0] = bbox[0][0];
            childBbox1[0][1] = bbox[0][1];
            childBbox1[0][2] = bbox[0][2];
            childBbox1[1][0] = childBbox1[0][0] + ((0 == maxDim) ? (median - childBbox1[0][0]) : halfSize[0]);
            childBbox1[1][1] = childBbox1[0][1] + ((1 == maxDim) ? (median - childBbox1[0][1]) : halfSize[1]);
            childBbox1[1][2] = childBbox1[0][2] + ((2 == maxDim) ? (median - childBbox1[0][2]) : halfSize[2]);

            glm::vec3 childBbox2[2];
            childBbox2[1][0] = bbox[1][0];
            childBbox2[1][1] = bbox[1][1];
            childBbox2[1][2] = bbox[1][2];
            childBbox2[0][0] = childBbox1[0][0] + ((0 == maxDim) ? (median - childBbox1[0][0]) : 0);
            childBbox2[0][1] = childBbox1[0][1] + ((1 == maxDim) ? (median - childBbox1[0][1]) : 0);
            childBbox2[0][2] = childBbox1[0][2] + ((2 == maxDim) ? (median - childBbox1[0][2]) : 0);

            for (int i = 0; i < 2; i++)
            {
                HybridVH *child = new HybridVH((i == 0) ? childBbox1 : childBbox2);
                // See which of the splats go into the newly created node
                for (int k = 0; k < containedSplats->size(); k++)
                {
                    auto splat = (*containedSplats)[k];
                    if (insideBBox(child->bbox, splat, sd))
                    {
                        child->containedSplats->push_back(splat);
                        addSplatToCoverage(child->coverage, splat, sd);
                    }
                }

                children.push_back(child);
            }
        } 
        
        containedSplats->clear();
        
        for(auto child : children){
            if (level < MAX_HYBRID_LEVEL - 1)
            {
                child->isLeaf = false;
                child->processSplats(level + 1, sd, progress);
            }
            else
            {
                child->isLeaf = true;
                child->level = MAX_HYBRID_LEVEL;
                computeNodeRepresentative(child, sd);
            } 
        }
    }
}

HybridVH::~HybridVH()
{
    for (auto child : children)
        if (child != nullptr)
        {
            delete child;
        }
    delete containedSplats;
}

void HybridVH::buildVHStructure(std::vector<SplatData> &sd, uint32_t num_primitives, volatile int *progress)
{
    glm::vec3 minBound(1e13, 1e13, 1e13);
    glm::vec3 maxBound(-1e13, -1e13, -1e13);

    for (int i = 0; i < num_primitives; i++)
    {
        minBound.x = min(minBound.x, sd[i].fields.position[0]);
        minBound.y = min(minBound.y, sd[i].fields.position[1]);
        minBound.z = min(minBound.z, sd[i].fields.position[2]);
        maxBound.x = max(maxBound.x, sd[i].fields.position[0]);
        maxBound.y = max(maxBound.y, sd[i].fields.position[1]);
        maxBound.z = max(maxBound.z, sd[i].fields.position[2]);
    }

    glm::vec3 center = (minBound + maxBound) * 0.5f;

    float maxSpan = max(maxBound.x - minBound.x, max(maxBound.y - minBound.y, maxBound.z - minBound.z));

    this->bbox[0] = center - maxSpan;
    this->bbox[1] = center + maxSpan;

    for (int i = 0; i < num_primitives; i++)
        this->containedSplats->push_back(i);

    this->processSplats(0, sd, progress);

    *progress = 1024;
}

int HybridVH::markForRender(bool *renderMask, uint32_t num_primitives, int renderLevel, glm::vec3 &cameraPosition, float fovy, int SW, float dpt)
{

    if (renderLevel == -1)
    {
        int shouldRenderNode = 0;
        if (this == nullptr)
            return 0;
        /* Easiest implementation, maximum projection by distance */
        float S = glm::length(this->coverage[0] - this->coverage[1]);
        float D = glm::length((this->coverage[0] + this->coverage[1]) / 2.0f - cameraPosition);

        float P = S / D * (SW / fovy);

        shouldRenderNode = (P > dpt);

        if (shouldRenderNode)
        { // is node big enough on the screen?
            if (this->isLeaf && this->containedSplats->size() > 0)
            {
                for (auto splat : *(this->containedSplats))
                    renderMask[splat] = true;
                return this->containedSplats->size();
            }
            else
            {
                int splatsRendered = 0;
                for (auto child : this->children)
                {
                    if (child != nullptr)
                        splatsRendered += child->markForRender(renderMask, num_primitives, renderLevel, cameraPosition, fovy, SW, dpt);
                }
                return splatsRendered;
            }
        }
        else
        {
            if (this->representative != 0)
            {
                renderMask[this->representative] = true;
                return 1;
            }
            else
            { // Level too low to have a representative, still have to go down
                int splatsRendered = 0;
                for (auto child : this->children)
                {
                    if (child != nullptr)
                        splatsRendered += child->markForRender(renderMask, num_primitives, renderLevel, cameraPosition, fovy, SW, dpt);
                }
                return splatsRendered;
            }
        }
    }
    else
    {
        if (this->level == renderLevel && this->representative != 0)
        {
            renderMask[this->representative] = true;
            return 1;
        }
        if (this->level < renderLevel && this->isLeaf)
        {
            for (auto splat : *(this->containedSplats)){
                renderMask[splat] = true;
            }
            return this->containedSplats->size();
        }
        if (!this->isLeaf && this->level < renderLevel)
        {
            int splatsRendered = 0;
            for (HybridVH *child : this->children)
            {
                if (child != nullptr)
                    splatsRendered += child->markForRender(renderMask, num_primitives, renderLevel, cameraPosition, fovy, SW, dpt);
            }
            return splatsRendered;
        }
        if (this->containedSplats->size() == 0 && this->representative == 0)
        {
            return 0;
        }
    }

    return 0;
}

#endif