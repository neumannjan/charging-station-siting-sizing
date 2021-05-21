#ifndef LOG_H_WULAQA8E
#define LOG_H_WULAQA8E

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <parallel_hashmap/phmap.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

typedef std::int64_t ord_t;
typedef std::int64_t idx_t;
typedef std::int16_t station_t;
typedef std::int32_t vehicle_t;
typedef std::int8_t attempt_t;
typedef std::int64_t timestamp_t;
typedef std::int32_t budget_t;

typedef std::int64_t total_t;

struct LogRow {
  idx_t idx;
  timestamp_t timestamp;
  vehicle_t vehicle;
  station_t station;
  std::double_t station_distance;
  std::double_t penalty;
  attempt_t attempt_no;
  std::int8_t status;

  LogRow(const idx_t idx, const timestamp_t timestamp, const vehicle_t vehicle,
         const station_t station, const std::double_t station_distance,
         const std::double_t penalty, const std::int8_t attempt_no,
         const std::int8_t status)
      : idx(idx), timestamp(timestamp), vehicle(vehicle), station(station),
        station_distance(station_distance), penalty(penalty),
        attempt_no(attempt_no), status(status) {}
};

struct TrafficRow {
  idx_t idx;
  timestamp_t arrival;
  timestamp_t departure;
  vehicle_t vehicle;
  station_t station;
  std::double_t station_distance;
  std::double_t penalty;
  attempt_t attempt_no;

  TrafficRow(const idx_t idx, const timestamp_t arrival,
             const timestamp_t departure, const vehicle_t vehicle,
             const station_t station, const std::double_t station_distance,
             const std::double_t penalty, const std::int8_t attempt_no)
      : idx(idx), arrival(arrival), departure(departure), vehicle(vehicle),
        station(station), station_distance(station_distance), penalty(penalty),
        attempt_no(attempt_no) {}
};

station_t find_max_station(const py::array_t<LogRow> &log);

inline station_t find_total_stations(const py::array_t<LogRow> &log) {
  return find_max_station(log) + 1;
}

idx_t find_max_idx(const py::array_t<LogRow> &log);

phmap::parallel_flat_hash_map<idx_t, attempt_t>
find_last_attempt_per_idx(const py::array_t<LogRow> &log);

total_t find_total_idx(const py::array_t<LogRow> &log);

total_t find_total_idx(const py::array_t<LogRow> &log,
                       const std::vector<station_t> &stations);

#endif /* end of include guard: LOG_H_WULAQA8E */
