#ifndef SIMULATION_H_HIOKGH18
#define SIMULATION_H_HIOKGH18

#include "log.h"
#include <iostream>
#include <type_traits>

namespace py = pybind11;

void assert_eligible_for_simulation(
    const py::array_t<LogRow> &log,
    const std::vector<budget_t> &budget_distribution);

struct SimulationVisitor {
  inline void visit_satisfied(idx_t idx, ord_t ord, attempt_t attempt,
                              station_t station) {}

  inline void visit_satisfied_upper_limit(idx_t idx) {}

  inline void visit_rejected(idx_t idx, ord_t ord, attempt_t attempt,
                             station_t station) {}

  inline static constexpr bool compute_upper_limit() { return false; }

  inline void finalize(std::vector<station_t> satisfied_stations) {}
};

template <typename visitor_t, bool update_budget_distribution = false>
void traverse_simulation_unchecked(
    visitor_t &visitor, const py::array_t<LogRow> &log,
    typename std::conditional<update_budget_distribution, std::vector<budget_t>,
                              const std::vector<budget_t>>::type
        &budget_distribution,
    const double max_station_distance, std::optional<idx_t> max_idx = {},
    std::optional<std::vector<budget_t>> upper_limit_distribution = {}) {
  auto it = log.unchecked<1>();

  const idx_t midx = max_idx ? *max_idx : find_max_idx(log);

  std::vector<station_t> satisfied_stations(midx + 1, -1);

  std::vector<budget_t> cumsum_diff = budget_distribution;
  std::vector<budget_t> max_cumsum_vals(budget_distribution.size(), 0);

  for (py::ssize_t i = 0; i < log.size(); ++i) {
    const LogRow &row = it(i);

    if (row.status == 1) {
      if (row.attempt_no == 0 && row.station_distance > max_station_distance) {
        // mark idx as not satisfiable FOR ANY ATTEMPT and skip
        satisfied_stations[row.idx] = -2; // must be different value than -1!
        visitor.visit_rejected(row.idx, i, row.attempt_no, row.station);
      } else if (satisfied_stations[row.idx] == -1 &&
                 cumsum_diff[row.station] > 0) {
        // satisfied_station MUST be equal to -1
        // (otherwise it is satisfied for a different attempt or fully
        // unsatisfiable)

        // mark idx as satisfied by this station
        satisfied_stations[row.idx] = row.station;
        visitor.visit_satisfied(row.idx, i, row.attempt_no, row.station);
      } else {
        // is either fully unsatisfiable or satisfied for a different attempt
        visitor.visit_rejected(row.idx, i, row.attempt_no, row.station);
      }
    }

    if (satisfied_stations[row.idx] == row.station) {
      // status added to cumsum IFF row.station is the same as the satisfied
      // station
      cumsum_diff[row.station] -= row.status;
      if constexpr (update_budget_distribution) {
        if (row.status == 1) {
          auto cumsum =
              budget_distribution[row.station] - cumsum_diff[row.station];

          if (cumsum > max_cumsum_vals[row.station])
            max_cumsum_vals[row.station] = cumsum;
        }
      }
    }
  }

  if constexpr (visitor_t::compute_upper_limit()) {
    if (!upper_limit_distribution)
      throw std::invalid_argument(
          "Simulation error: Empty upper_limit_distribution when "
          "compute_upper_limit() == true");

    std::vector<budget_t> &cumsums_upper_limit = *upper_limit_distribution;
    std::vector<bool> satisfied_upper_limit(midx + 1, false);

    for (py::ssize_t i = 0; i < log.size(); ++i) {
      const LogRow &row = it(i);

      if (satisfied_stations[row.idx] == -1) {
        if (row.status == 1 && cumsums_upper_limit[row.station] > 0 &&
            !satisfied_upper_limit[row.idx]) {
          satisfied_upper_limit[row.idx] = true;
          visitor.visit_satisfied_upper_limit(row.idx);
        }

        if (satisfied_upper_limit[row.idx])
          cumsums_upper_limit[row.station] -= row.status;
      }
    }
  }

  if constexpr (update_budget_distribution) {
    budget_distribution = max_cumsum_vals;
  }

  visitor.finalize(std::move(satisfied_stations));
}

template <bool do_compute_upper_limit>
struct _SatisfiedCountVisitor : public SimulationVisitor {
  total_t n_satisfied = 0;
  total_t n_satisfied_upper_limit = 0;

  inline void visit_satisfied(idx_t idx, ord_t ord, attempt_t attempt,
                              station_t station) {
    ++n_satisfied;
  }

  inline void visit_satisfied_upper_limit(idx_t idx) {
    if constexpr (do_compute_upper_limit) {
      ++n_satisfied_upper_limit;
    }
  }

  inline static constexpr bool compute_upper_limit() {
    return do_compute_upper_limit;
  }
};

template <bool update_budget_distribution = false>
inline total_t simulate_satisfied_unchecked(
    const py::array_t<LogRow> &log,
    typename std::conditional<update_budget_distribution, std::vector<budget_t>,
                              const std::vector<budget_t>>::type
        &budget_distribution,
    const double max_station_distance, std::optional<idx_t> max_idx) {
  _SatisfiedCountVisitor<false> vis;
  traverse_simulation_unchecked<decltype(vis), update_budget_distribution>(
      vis, log, budget_distribution, max_station_distance, max_idx);

  return vis.n_satisfied;
}

std::pair<total_t, total_t> simulate_satisfied_with_upper_limit_unchecked(
    const py::array_t<LogRow> &log, std::vector<budget_t> budget_distribution,
    std::vector<budget_t> upper_limit_additional_distribution,
    const double max_station_distance, std::optional<idx_t> max_idx = {});

py::array_t<station_t> simulate(const py::array_t<LogRow> &log,
                                std::vector<budget_t> budget_distribution,
                                const double max_station_distance,
                                std::optional<idx_t> max_idx = {});

#endif /* end of include guard: SIMULATION_H_HIOKGH18 */
