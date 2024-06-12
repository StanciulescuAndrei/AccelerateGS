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