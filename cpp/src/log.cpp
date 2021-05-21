#include "log.h"
#include <algorithm>
#include <utility>

station_t find_max_station(const py::array_t<LogRow> &log) {
  station_t max_val = -1;

  auto it = log.unchecked<1>();

  for (py::ssize_t i = 0; i < log.size(); ++i) {
    const LogRow &row = it(i);

    if (row.station > max_val)
      max_val = row.station;
  }

  return max_val;
}

idx_t find_max_idx(const py::array_t<LogRow> &log) {
  idx_t max_val = -1;

  auto it = log.unchecked<1>();

  for (py::ssize_t i = 0; i < log.size(); ++i) {
    const LogRow &row = it(i);

    if (row.idx > max_val)
      max_val = row.idx;
  }

  return max_val;
}

phmap::parallel_flat_hash_map<idx_t, attempt_t>
find_last_attempt_per_idx(const py::array_t<LogRow> &log) {
  phmap::parallel_flat_hash_map<idx_t, attempt_t> result;

  auto it = log.unchecked<1>();

  for (py::ssize_t i = 0; i < log.size(); ++i) {
    const LogRow &row = it(i);

    if (row.status == 1)
      result[row.idx] = row.attempt_no;
  }

  return result;
}

total_t find_total_idx(const py::array_t<LogRow> &log) {
  phmap::parallel_flat_hash_set<idx_t> idx_set;

  auto it = log.unchecked<1>();

  for (py::ssize_t i = 0; i < log.size(); ++i) {
    const LogRow &row = it(i);

    if (row.status == 1)
      idx_set.insert(row.idx);
  }

  return idx_set.size();
}

total_t find_total_idx(const py::array_t<LogRow> &log,
                       const std::vector<station_t> &stations) {
  phmap::parallel_flat_hash_set<idx_t> idx_set;
  phmap::flat_hash_set<station_t> stations_set;

  for(station_t st : stations) {
    stations_set.insert(st);
  }

  auto it = log.unchecked<1>();

  for (py::ssize_t i = 0; i < log.size(); ++i) {
    const LogRow &row = it(i);

    if (stations_set.contains(row.station))
      idx_set.insert(row.idx);
  }

  return idx_set.size();
}
