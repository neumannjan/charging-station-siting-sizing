#include "attempts.h"

timestamp_t distance_to_time_none(double distance) { return 0; }

std::vector<double>
distances_to_likelihood_quad(std::vector<double> distances) {
  for (int i = 0; i < distances.size(); ++i)
    distances[i] = 1 / (distances[i] * distances[i]);
  return distances;
}

py::array_t<TrafficRow> add_attempts_to_traffic(
    const py::array_t<TrafficRow> &traffic_npy,
    const std::vector<std::vector<std::double_t>> &station_distances_mtx,
    const int max_attempts, const double max_stations_pair_distance,
    const bool station_choice_probabilistic,
    const std::function<timestamp_t(double)> distance_to_time,
    const std::function<std::vector<double>(std::vector<double>)>
        distances_to_likelihood) {

  if (traffic_npy.ndim() != 1) {
    throw std::invalid_argument("traffic must be a 1D array");
  }

  std::vector<TrafficRow> traffic_new;
  traffic_new.reserve(traffic_npy.size() * max_attempts);

  auto traffic_it = traffic_npy.unchecked<1>();
  {
    std::vector<std::vector<std::double_t>> likelihoods_mtx(
        station_distances_mtx.size());

    for (int i = 0; i < station_distances_mtx.size(); ++i) {
      if (station_distances_mtx[i].size() != station_distances_mtx.size())
        throw std::invalid_argument(
            "station_distances_mtx must be a square matrix");

      likelihoods_mtx[i] = distances_to_likelihood(station_distances_mtx[i]);
    }

    std::vector<bool> accepted(station_distances_mtx.size());
    station_t st_previous;
    for (int t = 0; t < traffic_npy.size(); ++t) {
      const TrafficRow &row = traffic_it(t);

      if (row.penalty != 0)
        throw std::invalid_argument(
            "Traffic data already contains rows where penalty != 0.");

      if (row.attempt_no != 0)
        throw std::invalid_argument(
            "Traffic data already contains rows where attempt_no != 0.");

      timestamp_t time_offset = 0;
      double additional_distance = 0;
      st_previous = row.station;

      for (int i = 0; i < accepted.size(); ++i)
        accepted[i] = false;

      accepted[st_previous] = true;

      for (int i = 1; i < max_attempts; ++i) {
        const auto &dists = station_distances_mtx[st_previous];
        const auto &likelihoods = likelihoods_mtx[st_previous];

        bool found = false;

        int st_next;
        if (!station_choice_probabilistic) {
          double min_dist = std::numeric_limits<double>::max();

          for (int j = 0; j < dists.size(); ++j) {
            if (accepted[j] || dists[j] > max_stations_pair_distance)
              continue;

            if (dists[j] > 0 &&
                dists[j] != std::numeric_limits<double>::max() &&
                dists[j] < min_dist) {
              if (!found)
                found = true;
              min_dist = dists[j];
              st_next = j;
            }
          }
        } else {
          double sum = 0;
          for (int j = 0; j < dists.size(); ++j) {
            if (accepted[j] || dists[j] > max_stations_pair_distance ||
                likelihoods[j] != likelihoods[j])
              continue;

            sum += likelihoods[j];
          }

          double acc = 0;
          const double randval = ((double)rand() / (RAND_MAX));
          for (int j = 0; j < dists.size(); ++j) {
            if (accepted[j] || dists[j] > max_stations_pair_distance ||
                likelihoods[j] != likelihoods[j])
              continue;

            acc += likelihoods[j] / sum;
            if (acc > randval) {
              st_next = j;
              found = true;
              break;
            }
          }
        }

        if (!found)
          break;

        accepted[st_next] = true;
        time_offset += distance_to_time(dists[st_next]);
        additional_distance += dists[st_next];

        traffic_new.push_back(TrafficRow(
            row.idx, row.arrival + time_offset, row.departure + time_offset,
            row.vehicle, st_next, row.station_distance + additional_distance,
            additional_distance, i));
      }
    }
  }

  traffic_new.reserve(traffic_npy.size() + traffic_new.size());
  for (int i = 0; i < traffic_npy.size(); ++i) {
    TrafficRow row = traffic_it(i);
    traffic_new.push_back(row);
  }

  return py::array_t<TrafficRow>(traffic_new.size(), traffic_new.data());
}
