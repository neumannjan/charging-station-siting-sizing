#include <cstdlib>
#include <limits>
#include <numeric>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#include "attempts.h"
#include "distribution.h"
#include "simulation.h"

namespace py = pybind11;

PYBIND11_MODULE(fleet_electrification_cpp, m) {

  PYBIND11_NUMPY_DTYPE(LogRow, idx, timestamp, vehicle, station,
                       station_distance, penalty, attempt_no, status);

  PYBIND11_NUMPY_DTYPE(TrafficRow, idx, arrival, departure, vehicle, station,
                       station_distance, penalty, attempt_no);

  m.def("simulate", &simulate, "Simulate given traffic log", py::arg("log"),
        py::arg("budget_distribution"), py::arg("max_station_distance"),
        py::arg("max_idx") = std::optional<idx_t>());

  m.def("add_attempts_to_traffic", &add_attempts_to_traffic, py::arg("traffic"),
        py::arg("station_distances_mtx"), py::arg("max_attempts"),
        py::arg("max_stations_pair_distance"),
        py::arg("station_choice_probabilistic"), py::arg("distance_to_time"),
        py::arg("distances_to_likelihood"));
}
