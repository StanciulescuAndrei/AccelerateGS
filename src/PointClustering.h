#ifndef __POINT_CLUSTERING__
#define __POINT_CLUSTERING__

#include "oneapi/dal/algo/dbscan.hpp"
#include "oneapi/dal/io/csv.hpp"

namespace dal = oneapi::dal;

void DBSCANClustering(){
    const auto x_data = dal::read<dal::table>(dal::csv::data_source{ nullptr });

    double epsilon = 0.04;
    std::int64_t min_observations = 45;
    auto dbscan_desc = dal::dbscan::descriptor<>(epsilon, min_observations);
    dbscan_desc.set_result_options(dal::dbscan::result_options::responses);

    const auto result_compute = dal::compute(dbscan_desc, x_data);
}

#endif