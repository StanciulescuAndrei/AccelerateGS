#ifndef __POINT_CLUSTERING__
#define __POINT_CLUSTERING__

#include "daal.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include "raster_helper.cuh"

namespace dm = daal::data_management;
namespace algo =  daal::algorithms;

float dbscan_epsilon = 1e-3;

int SpectralClustering(glm::vec3 * bbox, std::vector<uint32_t> & containedSplatIds, std::vector<SplatData> & sd, int & nClusters, std::vector<int> & assignment){
    size_t inputDataSize = containedSplatIds.size();

    float boxSize = glm::length(bbox[1] - bbox[0]);
    int nIterations = 8;

    /* Transfer point data to float array */
    float * data = new float[3 * inputDataSize];
    for(int i = 0; i < inputDataSize; i++){
        data[i * 3 + 0] = sd[containedSplatIds[i]].fields.position[0];
        data[i * 3 + 1] = sd[containedSplatIds[i]].fields.position[1];
        data[i * 3 + 2] = sd[containedSplatIds[i]].fields.position[2];
    }

    // int nSamples = std::min(100, inputDataSize);

    // std::vector<int> indices(inputDataSize);
    // for (int i = 0; i < inputDataSize; ++i) {
    //     indices[i] = i;
    // }

    // std::srand(std::time(0));

    // std::random_shuffle(indices.begin(), indices.end());

    // indices.resize(nSamples);

    /*  Only do the spectral clustering part when there are few points.
        Otherwise, the eigenvector computation fo spectral decomposition 
        takes waay to long and requires too much RAM to be reasonable
    */
    if(inputDataSize < 32){
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
                    A(i, j) = glm::exp(-glm::distance(glm::make_vec3(data + i * 3), glm::make_vec3(data + j * 3)) / 8);
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
            data[i * 3 + 0] = eigenPairs[0].second(i);
            data[i * 3 + 1] = eigenPairs[1].second(i);
            data[i * 3 + 2] = eigenPairs[2].second(i);
        }

    }

    
    /* Move data to a oneDAL numeric table */
    dm::NumericTablePtr pointData = dm::HomogenNumericTable<>::create(data, 3, inputDataSize);

    /* Get k-means initialization points */
    algo::kmeans::init::Batch<float, algo::kmeans::init::randomDense> init(nClusters);
    init.input.set(algo::kmeans::init::data, pointData);

    init.compute();

    dm::NumericTablePtr centroids = init.getResult()->get(algo::kmeans::init::centroids);

    /* Create an algorithm object for the K-Means algorithm */
    algo::kmeans::Batch<> algorithm(nClusters, nIterations);

    algorithm.input.set(algo::kmeans::data, pointData);
    algorithm.input.set(algo::kmeans::inputCentroids, centroids);

    algorithm.parameter().resultsToEvaluate = algo::kmeans::computeAssignments |
                                              algo::kmeans::computeExactObjectiveFunction;

    algorithm.compute();

    /* Cleanup, then retrieve results */
    delete [] data;

    dm::NumericTablePtr assignmentResult;

    assignmentResult = algorithm.getResult()->get(algo::kmeans::assignments);
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