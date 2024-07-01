#ifndef __POINT_CLUSTERING__
#define __POINT_CLUSTERING__

#include "daal.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include "raster_helper.cuh"
#include "GUIManager.h"
#include <memory>

// #include <torch/torch.h>

namespace dm = daal::data_management;
namespace algo =  daal::algorithms;

void KMeansClustering(float * pointData, int dataSize, int numFeatures, std::vector<int> & assignment){
    /* The number of point features is equal to the number of requested clusters (after spectral reprojection) */
    int n_iter = 12;
    
    float * means = new float[numFeatures * numFeatures];

    /* Initialize means to first points in the data */
    for(int i = 0; i < numFeatures * numFeatures; i++){
        means[i] = pointData[i];
    }

    for(int iteration = 0; iteration < n_iter; iteration++){
        /* Find current assignments */
        for(int point = 0; point < dataSize; point++){
            float best_distance = 1e13;
            size_t best_cluster = 0;
            for(int cluster = 0; cluster < numFeatures; cluster++){
                float crt_distance = 0.0f;
                for(int feat = 0; feat < numFeatures; feat++){
                    crt_distance += std::pow(pointData[point * numFeatures + feat] - means[cluster * numFeatures + feat], 2);
                }
                if(crt_distance < best_distance){
                    best_distance = crt_distance;
                    best_cluster = cluster;
                }
            }
            assignment[point] = best_cluster;
        }

        /* Recompute means for new assignments */
        std::vector<size_t> counts(numFeatures, 0);
        for(int i = 0; i < numFeatures * numFeatures; i++){
            means[i] = 0.0f;
        }
        for(int point = 0; point < dataSize; point++){
            const size_t cluster = assignment[point];
            counts[cluster]++;
            for(int feat = 0; feat < numFeatures; feat++){
                means[cluster * numFeatures + feat] += pointData[point * numFeatures + feat];
            }
        }

        for(int cluster = 0; cluster < numFeatures; cluster++){
            const size_t div = std::max((size_t)1, counts[cluster]);
            for(int feat = 0; feat < numFeatures; feat++){
                means[cluster * numFeatures + feat] /= div;
            }
        }
    }

    delete [] means;

}

int PCReprojectionClustering(glm::vec3 * bbox, std::vector<uint32_t> & containedSplatIds, std::vector<SplatData> & sd, int & nClusters, std::vector<int> & assignment){
    size_t inputDataSize = containedSplatIds.size();
    int nIterations = 64;
    const int numFeatures = renderConfig.numClusterFeatures;

    /* Determine centers space bounds for centering and normalization */
    glm::vec3 minBound(1e13, 1e13, 1e13);
    glm::vec3 maxBound(-1e13, -1e13, -1e13);

    for (auto splat : containedSplatIds)
    {
        minBound.x = min(minBound.x, sd[splat].fields.position[0]);
        minBound.y = min(minBound.y, sd[splat].fields.position[1]);
        minBound.z = min(minBound.z, sd[splat].fields.position[2]);
        maxBound.x = max(maxBound.x, sd[splat].fields.position[0]);
        maxBound.y = max(maxBound.y, sd[splat].fields.position[1]);
        maxBound.z = max(maxBound.z, sd[splat].fields.position[2]);
    }

    float boxSize = glm::length(maxBound - minBound);

    Eigen::VectorXf meanPosition(numFeatures);
    for(int i = 0; i < numFeatures; i++){
        meanPosition(i) = 0.0f;
    }

    Eigen::MatrixXf pointFeatures(inputDataSize, numFeatures);

    /* Transfer point data to float array */
    float * data = new float[numFeatures * inputDataSize];

    float * reprojectedPointsData = new float[inputDataSize * nClusters];

    const float fieldScaling = 0.9f;
    for(int i = 0; i < inputDataSize; i++){
        data[i * numFeatures + 0] = sd[containedSplatIds[i]].fields.position[0];
        data[i * numFeatures + 1] = sd[containedSplatIds[i]].fields.position[1];
        data[i * numFeatures + 2] = sd[containedSplatIds[i]].fields.position[2];
        if(renderConfig.numClusterFeatures >= 6){
            data[i * numFeatures + 3] = sd[containedSplatIds[i]].fields.SH[0]   * boxSize * fieldScaling;
            data[i * numFeatures + 4] = sd[containedSplatIds[i]].fields.SH[1]   * boxSize * fieldScaling;  
            data[i * numFeatures + 5] = sd[containedSplatIds[i]].fields.SH[2]   * boxSize * fieldScaling;  
        }
        if(renderConfig.numClusterFeatures == 7){
            data[i * numFeatures + 6] = sd[containedSplatIds[i]].fields.opacity * boxSize * fieldScaling;  
        }
    }

    /* Compute average feature vector */
    for(int i = 0; i < inputDataSize; i++){
        for(int j = 0; j < numFeatures; j++){
            meanPosition(j) += data[i * numFeatures + j];
        }
    }
    meanPosition /= inputDataSize;

    /* Center feature points around mean */
    for(int i = 0; i < inputDataSize; i++){
        for(int j = 0; j < numFeatures; j++){
            pointFeatures(i, j) = data[i * numFeatures + j] - meanPosition(j);
        }
    }

    /* Compute covariance matrix of feature points */
    Eigen::MatrixXf covarianceMatrix(numFeatures, numFeatures);
    covarianceMatrix = pointFeatures.transpose() * pointFeatures;

    /* Compute PCA decomposition to determine the highest importance principal directions */
    Eigen::EigenSolver<Eigen::MatrixXf> eigensolver(covarianceMatrix);
    if (eigensolver.info() != Eigen::Success)
    {
        printf("Error in eigen-decomposition for PC Reprojection!!!\n");
    }

    Eigen::VectorXf eigenvalues = eigensolver.eigenvalues().real();
    Eigen::MatrixXf eigenvectors = eigensolver.eigenvectors().real();

    /* Deal with negative eigenvalues */
    eigenvalues = eigenvalues.cwiseAbs();

    // Create a vector of pairs where first element of pair is eigenvalue and second is corresponding eigenvector
    std::vector<std::pair<float, Eigen::VectorXf>> eigenPairs;
    for (int i = 0; i < eigenvalues.size(); i++) {
        float eigenValue = eigenvalues(i);
        Eigen::VectorXf eigenVector = eigenvectors.col(i);
        eigenPairs.push_back(std::make_pair(eigenValue, eigenVector)); 
    }

    // Sort the eigenpairs in descending order of eigenvalues
    std::sort(eigenPairs.begin(), eigenPairs.end(), [](std::pair<float, Eigen::VectorXf> a, std::pair<float, Eigen::VectorXf> b){
        return a.first > b.first;
    });

    /* Normalize first nClusters eigenvectors */
    for(int i = 0; i < nClusters; i++){
        eigenPairs[i].second.normalize();
    }

    /* Change data to reprojected points */
    for(int i = 0; i < inputDataSize; i++){
        for(int k = 0; k < nClusters; k++){
            reprojectedPointsData[i * nClusters + k] = pointFeatures.row(i) * eigenPairs[k].second;
        }
    }

    KMeansClustering(reprojectedPointsData, inputDataSize, nClusters, assignment);
    return 0;

    /* Move data to a oneDAL numeric table */
    dm::NumericTablePtr pointData = dm::HomogenNumericTable<>::create(reprojectedPointsData, nClusters, inputDataSize);

    /* Get k-means initialization points */
    std::unique_ptr<algo::kmeans::init::Batch<float, algo::kmeans::init::randomDense>> init(new algo::kmeans::init::Batch<float, algo::kmeans::init::randomDense>(nClusters));
    init->input.set(algo::kmeans::init::data, pointData);

    init->compute();

    dm::NumericTablePtr centroids = init->getResult()->get(algo::kmeans::init::centroids);

    init->resetCompute();

    /* Create an algorithm object for the K-Means algorithm */
    std::unique_ptr<algo::kmeans::Batch<>> algorithm(new algo::kmeans::Batch<>(nClusters, nIterations));

    algorithm->input.set(algo::kmeans::data, pointData);
    algorithm->input.set(algo::kmeans::inputCentroids, centroids);

    algorithm->parameter().resultsToEvaluate = algo::kmeans::computeAssignments;

    algorithm->compute();

    /* Cleanup, then retrieve results */
    delete [] data;
    delete [] reprojectedPointsData; 

    dm::NumericTablePtr assignmentResult;

    assignmentResult = algorithm->getResult()->get(algo::kmeans::assignments);

    algorithm->resetCompute();
    
    /* Data extraction containers */
    dm::BlockDescriptor<int> block;
    int *array;

    /* If good number of clusters, get cluster assignment */
    size_t nRows = assignmentResult->getNumberOfRows();
    size_t nCols = assignmentResult->getNumberOfColumns();

    assignmentResult->getBlockOfRows(0, nRows, dm::ReadWriteMode::readOnly, block);
    array = block.getBlockPtr();
    for(int i = 0; i < nRows; i++){
        assignment.push_back(array[i]);
    }
    assignmentResult->releaseBlockOfRows(block);

    assignmentResult.reset();
    centroids.reset();
    pointData.reset();

    return 0;
}

int SpectralClustering(glm::vec3 * bbox, std::vector<uint32_t> & containedSplatIds, std::vector<SplatData> & sd, int & nClusters, std::vector<int> & assignment){
    size_t inputDataSize = containedSplatIds.size();

    /* Compute coverage spanned by splat centers */

    glm::vec3 minBound(1e13, 1e13, 1e13);
    glm::vec3 maxBound(-1e13, -1e13, -1e13);

    for (auto splat : containedSplatIds)
    {
        minBound.x = min(minBound.x, sd[splat].fields.position[0]);
        minBound.y = min(minBound.y, sd[splat].fields.position[1]);
        minBound.z = min(minBound.z, sd[splat].fields.position[2]);
        maxBound.x = max(maxBound.x, sd[splat].fields.position[0]);
        maxBound.y = max(maxBound.y, sd[splat].fields.position[1]);
        maxBound.z = max(maxBound.z, sd[splat].fields.position[2]);
    }

    float boxSize = glm::length(maxBound - minBound);

    glm::vec3 meanPosition(0.0f);
    for(auto splat : containedSplatIds){
        meanPosition += glm::make_vec3(sd[splat].fields.position);
    }

    meanPosition /= inputDataSize;

    int nIterations = 64;
    const int numFeatures = renderConfig.numClusterFeatures;

    /* Transfer point data to float array */
    float * data = new float[numFeatures * inputDataSize];
    const float fieldScaling = 1.0f;
    for(int i = 0; i < inputDataSize; i++){
        data[i * numFeatures + 0] = sd[containedSplatIds[i]].fields.position[0];
        data[i * numFeatures + 1] = sd[containedSplatIds[i]].fields.position[1];
        data[i * numFeatures + 2] = sd[containedSplatIds[i]].fields.position[2];
        if(renderConfig.numClusterFeatures == 7){
            data[i * numFeatures + 3] = sd[containedSplatIds[i]].fields.SH[0]   * boxSize * fieldScaling;
            data[i * numFeatures + 4] = sd[containedSplatIds[i]].fields.SH[1]   * boxSize * fieldScaling;  
            data[i * numFeatures + 5] = sd[containedSplatIds[i]].fields.SH[2]   * boxSize * fieldScaling;  
            data[i * numFeatures + 6] = sd[containedSplatIds[i]].fields.opacity * boxSize * fieldScaling;  
        }
        
    }

    /*  Only do the spectral clustering part when there are few points.
        Otherwise, the eigenvector computation fo spectral decomposition 
        takes waay to long and requires too much RAM to be reasonable
    */
    bool reprojectPoints = (inputDataSize < renderConfig.spectralClusteringThreshold);

    float * reprojectedPointsData;
    if(reprojectPoints){
        reprojectedPointsData = new float[nClusters * inputDataSize];
    }
    else{
        reprojectedPointsData = data;
    }

    if(reprojectPoints){
        Eigen::MatrixXf A(inputDataSize, inputDataSize);
        Eigen::MatrixXf L(inputDataSize, inputDataSize);
        Eigen::DiagonalMatrix<float, Eigen::Dynamic> D(inputDataSize);

        /* Build affinity matrix */
        for(int i = 0; i < inputDataSize; i++){
            for(int j = 0; j < inputDataSize; j++){
                if(i == j){
                    A(i, j) = 0.0f;
                }
                else{
                    A(i, j) = glm::exp(-glm::distance(glm::make_vec3(data + i * numFeatures), glm::make_vec3(data + j * numFeatures)) / 8);
                }
            }
        }

        /* Build diagonal matrix and compute inverse square root */
        D.diagonal() = A.rowwise().sum();
        for(int i = 0; i < inputDataSize; i++){
            D.diagonal()[i] = 1 / glm::sqrt(D.diagonal()[i]);
        }

        /* Laplacian of the affinity matrix: L = D^(-1/2) * A * D^(-1/2) */
        L = D*A*D;
        Eigen::EigenSolver<Eigen::MatrixXf> eigensolver(L);

        if (eigensolver.info() != Eigen::Success)
        {
            printf("Error in eigen-decomposition for Spectral Clustering!!!\n");
        }

        Eigen::VectorXf eigenvalues = eigensolver.eigenvalues().real();
        Eigen::MatrixXf eigenvectors = eigensolver.eigenvectors().real();

        /* Deal with negative eigenvalues */
        eigenvalues = eigenvalues.cwiseAbs();

        // Create a vector of pairs where first element of pair is eigenvalue and second is corresponding eigenvector
        std::vector<std::pair<float, Eigen::VectorXf>> eigenPairs;
        for (int i = 0; i < eigenvalues.size(); i++) {
            float eigenValue = eigenvalues(i);
            Eigen::VectorXf eigenVector = eigenvectors.col(i);
            eigenPairs.push_back(std::make_pair(eigenValue, eigenVector)); 
        }

        // Sort the eigenpairs in descending order of eigenvalues
        std::sort(eigenPairs.begin(), eigenPairs.end(), [](std::pair<float, Eigen::VectorXf> a, std::pair<float, Eigen::VectorXf> b){
            return a.first > b.first;
        });

        /* Change data to reprojected points */
        for(int i = 0; i < inputDataSize; i++){
            for(int k = 0; k < nClusters; k++){
                reprojectedPointsData[i * nClusters + k] = eigenPairs[k].second(i);
            }
        }

    }

    /* Move data to a oneDAL numeric table */
    dm::NumericTablePtr pointData = dm::HomogenNumericTable<>::create(reprojectedPointsData, reprojectPoints ? nClusters : numFeatures, inputDataSize);

    /* Get k-means initialization points */
    algo::kmeans::init::Batch<float, algo::kmeans::init::randomDense> init(nClusters);
    init.input.set(algo::kmeans::init::data, pointData);

    init.compute();

    dm::NumericTablePtr centroids = init.getResult()->get(algo::kmeans::init::centroids);

    init.resetCompute();

    /* Create an algorithm object for the K-Means algorithm */
    algo::kmeans::Batch<> algorithm(nClusters, nIterations);

    algorithm.input.set(algo::kmeans::data, pointData);
    algorithm.input.set(algo::kmeans::inputCentroids, centroids);

    algorithm.parameter().resultsToEvaluate = algo::kmeans::computeAssignments |
                                              algo::kmeans::computeExactObjectiveFunction;

    algorithm.compute();

    /* Cleanup, then retrieve results */
    delete [] data;
    if(reprojectPoints){
        delete [] reprojectedPointsData; 
    }

    dm::NumericTablePtr assignmentResult;

    assignmentResult = algorithm.getResult()->get(algo::kmeans::assignments);

    algorithm.resetCompute();
    /* Data extraction containers */
    dm::BlockDescriptor<int> block;
    int *array;

    /* If good number of clusters, get cluster assignment */
    size_t nRows = assignmentResult->getNumberOfRows();
    size_t nCols = assignmentResult->getNumberOfColumns();

    assignmentResult->getBlockOfRows(0, nRows, dm::ReadWriteMode::readOnly, block);
    array = block.getBlockPtr();
    for(int i = 0; i < nRows; i++){
        assignment.push_back(array[i]);
    }
    assignmentResult->releaseBlockOfRows(block);

    assignmentResult.reset();
    pointData.reset();

    return 0;
}

int DBSCANClustering(glm::vec3 * bbox, std::vector<uint32_t> & containedSplatIds, std::vector<SplatData> & sd, int & nClusters, std::vector<int> & assignment){
    size_t inputDataSize = containedSplatIds.size();

    float boxSize = glm::length(bbox[1] - bbox[0]);

    /* Transfer point data to float array */
    float * data = new float[3 * inputDataSize];
    for(int i = 0; i < inputDataSize; i++){
        data[i * 3 + 0] = sd[containedSplatIds[i]].fields.position[0];
        data[i * 3 + 1] = sd[containedSplatIds[i]].fields.position[1];
        data[i * 3 + 2] = sd[containedSplatIds[i]].fields.position[2];
    }

    /* Move data to a oneDAL numeric table */
    dm::NumericTablePtr pointData = dm::HomogenNumericTable<>::create(data, 3, inputDataSize);

    dm::NumericTablePtr nclustersResult;
    dm::NumericTablePtr assignmenResult;

    /* Data extraction containers */
    dm::BlockDescriptor<int> block;
    int *array;

    float dynamicEpsilon = boxSize / 30.0f;

    
    int nTries = 0;

    while(nTries < 10){

        /* Set up DBSCAN algorithm handler */
        daal::algorithms::dbscan::Batch<> dbscanClusterer(dynamicEpsilon, 2);
        dbscanClusterer.input.set(daal::algorithms::dbscan::data, pointData);

        /* Compute clustering */
        dbscanClusterer.compute();

        nclustersResult = dbscanClusterer.getResult()->get(algo::dbscan::nClusters);
        assignmenResult = dbscanClusterer.getResult()->get(algo::dbscan::assignments);

        /* Get the number of identified clusters */
        nclustersResult->getBlockOfRows(0, 1, dm::ReadWriteMode::readOnly, block);
        array = block.getBlockPtr();

        nClusters = array[0];

        nclustersResult->releaseBlockOfRows(block);

        if(nClusters > 3){
            dynamicEpsilon *= 1.5f;
            nTries++;
        }
        else if(nClusters == 1){
            dynamicEpsilon /= 1.5f;
            nTries++;
        }
        else{
            break;
        }
        
    }

    if(nClusters < 2){
        return -1; // Not great, we would like 2/3 clusters, not 1 or 0
    }

    /* Cleanup, then retrieve results */
    delete [] data;
    
    /* If good number of clusters, get cluster assignment */

    size_t nRows = assignmenResult->getNumberOfRows();
    size_t nCols = assignmenResult->getNumberOfColumns();

    assignmenResult->getBlockOfRows(0, nRows, dm::ReadWriteMode::readOnly, block);

    array = block.getBlockPtr();
    for(int i = 0; i < nRows; i++){
        assignment[i] = array[i];
    }
    assignmenResult->releaseBlockOfRows(block);

    return 0;

}

#endif