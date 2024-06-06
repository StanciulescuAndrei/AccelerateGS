#ifndef __POINT_CLUSTERING__
#define __POINT_CLUSTERING__

#include "daal.h"

namespace dm = daal::data_management;
namespace algo =  daal::algorithms;

void DBSCANClustering(){
    float data[8] = {0.0f, 1.0f, 0.0f, 1.001f, 2.0f, 0.0f, 2.001f, 0.0f};

    dm::NumericTablePtr pointData = dm::HomogenNumericTable<>::create(data, 2, 4);

    daal::algorithms::dbscan::Batch<> dbscanClusterer(1e-3, 2);
    dbscanClusterer.input.set(daal::algorithms::dbscan::data, pointData);

    dbscanClusterer.compute();

    dm::NumericTablePtr result = dbscanClusterer.getResult()->get(algo::dbscan::assignments);

    size_t nRows = result->getNumberOfRows();
    size_t nCols = result->getNumberOfColumns();

    dm::BlockDescriptor<int> block;
    result->getBlockOfRows(0, nRows, dm::ReadWriteMode::readOnly, block);
    int *array = block.getBlockPtr();
    printf("%d, %d\n", nRows, nCols);
    for(int i=0;i<nRows;i++){
        printf("-> %d\n", array[i]);
    }
    printf("----------------\n");
    result->releaseBlockOfRows(block);

}

#endif