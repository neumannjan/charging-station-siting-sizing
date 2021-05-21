#ifndef HELPERS_H_ENSY2BF5
#define HELPERS_H_ENSY2BF5

#include "parallel_hashmap/phmap_utils.h"
#include <functional>
#include <sstream>
#include <string>

namespace std {

template <typename T> struct hash<std::vector<T>> {
  std::size_t operator()(const std::vector<T> &vec) const {
    std::size_t seed = vec.size();
    for (auto &i : vec) {
      seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

template <typename T1, typename T2> struct hash<std::pair<T1, T2>> {
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    return phmap::HashState().combine(0, p.first, p.second);
  }
};

} // namespace std

template <typename T> std::string print_value(const T &value) {
  return std::to_string(value);
}

template <typename T1, typename T2>
std::string print_value(const std::pair<T1, T2> &value);

template <typename T> std::string print_value(const std::vector<T> &value);

inline std::string print_depth(int value) {
  std::stringstream ss;
  for (int i = 0; i < value; ++i)
    ss << "| ";
  return ss.str();
}

template <typename T1, typename T2>
std::string print_value(const std::pair<T1, T2> &value) {
  std::stringstream ss;
  ss << print_value(value.first);
  ss << " -> ";
  ss << print_value(value.second);
  return ss.str();
}

template <typename iterator>
std::string print_iter(const iterator begin, const iterator end) {
  std::stringstream ss;
  ss << "[ ";
  for (auto it = begin; it != end; ++it) {
    ss << print_value(*it) << ", ";
  }
  ss << ']';
  return ss.str();
}

template <typename T> std::string print_value(const std::vector<T> &value) {
  return print_iter(value.begin(), value.end());
}

template <typename iterator, typename set_t = typename iterator::value_type>
set_t set_union_all(const iterator begin, const iterator end) {
  if (begin == end)
    return set_t();

  auto it = begin;
  set_t retval = *it++;

  for (; it != end; ++it) {
    for (auto &v : *it) {
      retval.insert(v);
    }
  }

  return retval;
}

template <typename set_t> set_t set_union(set_t set1, const set_t &set2) {
  for (auto &v : set2)
    set1.insert(v);

  return set1;
}

template <typename T>
std::vector<T> vector_concat(std::vector<T> vec1, const std::vector<T> &vec2) {
  for (auto &v : vec2)
    vec1.push_back(v);

  return vec1;
}

#endif /* end of include guard: HELPERS_H_ENSY2BF5 */
