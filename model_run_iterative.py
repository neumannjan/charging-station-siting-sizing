import argparse

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

import lib.budget as budget
import lib.simulation as sim
import lib.utils as utils
import lib.optimize.utils as optutils
from lib.optimize import build_soft_assign_model
from lib.optimize.soft_assign import _ModelBuilder, ModelSpec

parser = argparse.ArgumentParser()
parser.add_argument("spec_input_file")
parser.add_argument("output_file")
parser.add_argument("--cpus", type=int, required=False)

args = parser.parse_args()

spec: ModelSpec = utils.gzip_pickle_load(args.spec_input_file)

print()
print(args.spec_input_file)
print(args.output_file)
print("Stations:", sorted(spec.log.station.unique()))

print()
print("---")
print()

print(spec)

print()
print("---")
print()

with build_soft_assign_model(spec) as m:
    m.model.setParam(GRB.Param.MIPFocus, 1)
    m.model.setParam(GRB.Param.NodeMethod, 1)
    if args.cpus is not None:
        m.model.setParam(GRB.Param.Threads, args.cpus)

    plog = sim.PreprocessedLog.from_dataframe(spec.log)

    try:
        solutions = utils.gzip_pickle_load(args.output_file)
        if not optutils.is_wrapped(solutions):
            if not spec.first_come_first_served:
                raise Exception("non-FCFS solution must not be unwrapped!")

            solutions = optutils.wrap_distributions(solutions)(
                plog, max_station_distance=spec.max_station_distance)

        highest_budget_dist = solutions[max(solutions.keys())].distribution
        this_sol = solutions[min(solutions.keys())]
        next_dist = this_sol.distribution.copy()
        next_upper_bound = this_sol.objective
    except:
        result: optutils.Solution = m.optimize(
            return_as_solution=True, return_simulation=False, compute_iis=True)

        if m.model.getAttr(GRB.Attr.Status) != GRB.OPTIMAL:
            print("Not optimal.")
            print("Quitting early.")
            exit(1)

        print("Starter solution:", spec.starter_solution_distribution)
        print("Best solution:   ", result.distribution)
        print("Best objective:  ", result.objective)

        solutions = {result.budget: result}
        highest_budget_dist = result.distribution
        next_dist = result.distribution.copy()
        next_upper_bound = result.objective

        utils.gzip_pickle_dump(solutions, args.output_file)

    max_station = spec.log.station.max()
    max_idx = spec.log.index.max()

    for dist_sum in range(sum(next_dist) - 1, 0, -1):
        print()
        print("BUDGET:", dist_sum)
        print()

        results = [0 for _ in range(max_station + 1)]
        for i in range(max_station + 1):
            if next_dist[i] == 0:
                continue
            if next_dist[i] == 1 and spec.force_build_stations_distribution is not None and spec.force_build_stations_distribution[i] >= 1:
                continue

            next_dist[i] -= 1
            results[i] = sim.get_satisfied_charging_requests(sim.simulate(
                plog, next_dist, max_station_distance=spec.max_station_distance, max_station=max_station, max_idx=max_idx))
            next_dist[i] += 1

        next_dist[np.argmax(results)] -= 1

        m.primary_optim_coverage_for_given_budget(
            dist_sum, upper_bound=next_upper_bound)
        m.add_starter_solution(next_dist.copy())

        # add secondary objective only if the original spec has secondary objective
        if spec.secondary_opt_distance_sum or \
                spec.secondary_opt_min_stations or \
                spec.secondary_optim_closest_to_distribution is not None:
            m.secondary_optim_closest_to_dist(list(highest_budget_dist))

        result: optutils.Solution = m.optimize(
            return_as_solution=True, return_simulation=False, compute_iis=True)

        solutions[dist_sum] = result
        next_upper_bound = result.objective

        if m.model.getAttr(GRB.Attr.Status) != GRB.OPTIMAL:
            print("Not optimal.")
            print("Quitting early.")
            exit(1)

        utils.gzip_pickle_dump(solutions, args.output_file)

        print("Starter solution:", budget.ListBudgetDist(next_dist))
        next_dist = result.distribution.copy()
        print("Best solution:   ", budget.ListBudgetDist(next_dist))
        print("Best objective:  ", result.objective)
