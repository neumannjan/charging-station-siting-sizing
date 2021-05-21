#ifndef ATTEMPTS_H_EA7G60S3
#define ATTEMPTS_H_EA7G60S3

#include "log.h"

namespace py = pybind11;

std::int64_t distance_to_time_none(double distance);

std::vector<double> distances_to_likelihood_quad(std::vector<double> distances);

py::array_t<TrafficRow> add_attempts_to_traffic(
    const py::array_t<TrafficRow> &traffic_npy,
    const std::vector<std::vector<std::double_t>> &station_distances_mtx,
    const int max_attempts, const double max_stations_pair_distance,
    const bool station_choice_probabilistic,
    const std::function<timestamp_t(double)> distance_to_time,
    const std::function<std::vector<double>(std::vector<double>)>
        distances_to_likelihood);

#endif /* end of include guard: ATTEMPTS_H_EA7G60S3 */
