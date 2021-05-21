#include "simulation.h"
#include "helpers.h"
#include <utility>

void assert_eligible_for_simulation(
    const py::array_t<LogRow> &log,
    const std::vector<budget_t> &budget_distribution) {
  if (log.ndim() != 1)
    throw std::invalid_argument("Invalid number of dimensions of log");

  auto it = log.unchecked<1>();

  phmap::parallel_flat_hash_map<idx_t, attempt_t> last_seen_attempt;
  phmap::parallel_flat_hash_set<std::pair<idx_t, attempt_t>> arrived;

  for (py::ssize_t i = 0; i < log.size(); ++i) {
    const LogRow &row = it(i);
    if (static_cast<size_t>(row.station) >= budget_distribution.size())
      throw std::invalid_argument(
          std::string("budget_distribution has invalid size ") +
          std::to_string(budget_distribution.size()) +
          "(not enough stations). Station " + std::to_string(row.station) +
          " does not fit.");

    if (row.station < 0)
      throw std::invalid_argument(std::string("invalid station ") +
                                  std::to_string(row.station) +
                                  ". Must be >= 0.");

    if (row.attempt_no < 0)
      throw std::invalid_argument(std::string("invalid attempt_no ") +
                                  std::to_string(row.attempt_no) +
                                  ". Must be >= 0.");

    if (row.status == 1) {
      if (!arrived.insert(std::make_pair(row.idx, row.attempt_no)).second)
        throw std::invalid_argument(
            std::string("Multiple rows found with idx == ") +
            std::to_string(row.idx) +
            " && attempt_no == " + std::to_string(row.attempt_no));

      auto it = last_seen_attempt.find(row.idx);
      const attempt_t lsa = it == last_seen_attempt.end() ? -1 : it->second;

      if (row.attempt_no <= lsa)
        throw std::invalid_argument(std::string(
            "Attempt numbers must be non-negative and strictly monotonically "
            "increasing."));

      if (it == last_seen_attempt.end())
        last_seen_attempt.emplace(row.idx, row.attempt_no);
      else
        it->second = row.attempt_no;
    } else if (row.status == -1) {
      if (!arrived.contains(std::make_pair(row.idx, row.attempt_no)))
        throw std::invalid_argument(
            std::string("Row with idx == ") + std::to_string(row.idx) +
            " && attempt_no == " + std::to_string(row.attempt_no) +
            " departs before it arrives.");
    } else {
      throw std::invalid_argument(std::string("invalid status: ") +
                                  std::to_string(row.status) +
                                  ". Must be -1 or 1.");
    }
  }
}

struct SatisfiedStationsVisitor : public SimulationVisitor {
  std::vector<station_t> satisfied_stations;

  inline void finalize(std::vector<station_t> satisfied_stations) {
    this->satisfied_stations = std::move(satisfied_stations);
  }
};

py::array_t<station_t> simulate(const py::array_t<LogRow> &log,
                                std::vector<budget_t> budget_distribution,
                                const double max_station_distance,
                                std::optional<idx_t> max_idx) {
  assert_eligible_for_simulation(log, budget_distribution);

  SatisfiedStationsVisitor vis;
  traverse_simulation_unchecked(vis, log, budget_distribution,
                                max_station_distance, max_idx);

  return py::array_t<station_t>(vis.satisfied_stations.size(),
                                vis.satisfied_stations.data());
}

std::pair<total_t, total_t> simulate_satisfied_with_upper_limit_unchecked(
    const py::array_t<LogRow> &log, std::vector<budget_t> budget_distribution,
    std::vector<budget_t> upper_limit_additional_distribution,
    const double max_station_distance, std::optional<idx_t> max_idx) {
  _SatisfiedCountVisitor<true> vis;
  traverse_simulation_unchecked(vis, log, budget_distribution,
                                max_station_distance, max_idx,
                                upper_limit_additional_distribution);

  return {vis.n_satisfied, vis.n_satisfied + vis.n_satisfied_upper_limit};
}
