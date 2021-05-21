import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .. import simulation as sim
from .. import utils
from ..budget import BudgetDist, ListBudgetDist, DictBudgetDist
from .hard_assign import optimize_hard_assign
from .. import traffic as trf
from . import utils as optutils


def codify(vals):
    vals = np.array(vals)
    vals_map, codes = np.unique(vals, return_inverse=True)
    codes = np.reshape(codes, vals.shape)
    return np.array(vals_map), codes


def idx_groups(log: pd.DataFrame, additional_cols=[]):
    # creates an array of indices, sorted by unique element
    # sorts records array so all unique elements are together
    log_sorted = log[additional_cols].reset_index()
    log_sorted.sort_values(
        [log_sorted.columns[0], *additional_cols], inplace=True)

    idx_sort = np.array(log_sorted.index)
    sorted_vals = np.array(log_sorted[log_sorted.columns[0]])

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    _, idx_start = np.unique(
        sorted_vals, return_index=True)

    # splits the indices into separate arrays
    res = np.split(idx_sort, idx_start[1:])

    return res


class _ModelBuilder:

    def __init__(self):
        self.model: gp.Model = None
        self.log: pd.DataFrame = None
        self.log_original: pd.DataFrame = None
        self.max_station_distance: float = float('inf')
        self.max_allowed_budget: int = None
        self.first_come_first_served = True
        self.station_subset_only_mode = False
        self.debug = False

        self.reattempts_disallowed = False

        # variables
        self.xs: Any = None
        self.bs: Any = None
        self.bs_built: Any = None

        self.V_map: Any = None
        self.V: Any = None
        self.L_map: Any = None
        self.L: Any = None
        self.S: Any = None
        self.st_mask: Any = None
        self.A_groups: Any = None

        self.objective_constraints: Dict[int, List[Any]] = dict()
        self.forced_built_stations_constraints: List[Any] = []

    def enter(self):
        if self.model is None:
            self.model = gp.Model("station_budget_soft_assign")
        return self

    def __enter__(self):
        return self.enter()

    def __exit__(self, a, b, c):
        self.model.dispose()

    def set_debug(self, debug: bool):
        self.debug = debug
        return self

    def init_log(self, log: pd.DataFrame, max_allowed_budget: int):
        self.log_original = log
        self.log = log.copy()
        self.max_allowed_budget = max_allowed_budget

        if self.debug:
            print("Max allowed budget:", max_allowed_budget)
        return self

    def init_settings(self, station_subset_only_mode: bool = False):
        self.station_subset_only_mode = station_subset_only_mode
        return self

    def limit_station_distance(self, max_station_distance: float):
        if max_station_distance < float('inf'):
            if 'attempt_no' in self.log.columns:
                drop_idx = self.log.loc[(self.log['attempt_no'] == 0) & (
                    self.log['station_distance'] > max_station_distance)].index
            else:
                drop_idx = self.log.loc[self.log['station_distance']
                                        > max_station_distance].index

            pre = len(self.log)
            self.log.drop(drop_idx, inplace=True)
            if self.debug:
                print("Dropped", pre - len(self.log),
                      "rows due to station distance limit.")

        self.max_station_distance = max_station_distance
        return self

    def add_variables(self):
        self.V_map, self.V = codify(self.log['vehicle'])
        self.L_map, self.L = codify(self.log['station'])
        self.S = self.log['status'].tolist()

        if self.debug:
            print(f"Station mapping: 0 to {np.max(self.L)} (incl.) -",
                  {i: v for i, v in enumerate(self.L_map)})

        self.st_mask = self.log['status'] == 1

        self.A_groups = idx_groups(self.log, ['attempt_no'])

        if self.debug:
            for g in self.A_groups:
                b = True
                l = self.log.iloc[g]
                l_e = self.log.iloc[g[::2]]
                l_o = self.log.iloc[g[1::2]]
                b = b and (l.index == l.index[0]).all()
                b = b and (l_e['status'] == 1).all()
                b = b and (l_o['status'] == -1).all()
                b = b and (l_e.index == l_o.index).all()
                if not b:
                    print("Error in A_groups")
                    utils.display(l)

        if not self.station_subset_only_mode:
            self.bs = self.model.addMVar(shape=len(self.L_map), vtype=GRB.INTEGER,
                                         lb=0, ub=self.max_allowed_budget, name="b")
        self.xs = self.model.addMVar(shape=len(self.log),
                                     vtype=GRB.BINARY, name="x")
        self.bs_built = self.model.addMVar(
            shape=len(self.L_map), vtype=GRB.BINARY, name="bs_built")

        return self

    def add_common_constraints(self, first_come_first_served=True, progress=False):
        self.first_come_first_served = first_come_first_served

        xs_l = self.xs.tolist()
        bs_l = self.bs.tolist() if not self.station_subset_only_mode else None
        bs_built_l = self.bs_built.tolist()

        # built binary constraints
        if not self.station_subset_only_mode:
            for i in range(len(self.L_map)):
                self.model.addLConstr(
                    bs_built_l[i], GRB.LESS_EQUAL, bs_l[i], name=f"st-{self.L_map[i]}-built-lower")
                self.model.addLConstr(bs_l[i], GRB.LESS_EQUAL, self.max_allowed_budget *
                                      bs_built_l[i], name=f"st-{self.L_map[i]}-built-upper")

        # arrdep constraints
        for g in utils.progressify(self.A_groups, desc="Arrival/departure constraints", enabled=progress):
            for i in range(0, len(g), 2):
                self.model.addLConstr(xs_l[g[i]], GRB.EQUAL,
                                      xs_l[g[i+1]], f"arrdep_xs_{g[i]}_and_{g[i+1]}")

        # attempt constraints
        for g in utils.progressify(self.A_groups, desc="Attempt constraints", enabled=progress):
            self.model.addLConstr(
                sum(self.xs[g[::2]].tolist()), GRB.LESS_EQUAL, 1, f"attempts-xs-of-{'-'.join(str(v) for v in g[::2].tolist())}_<=_1")
            # self.model.addSOS(GRB.SOS_TYPE1, self.xs[g[::2]].tolist())

        if self.station_subset_only_mode:
            for i, l in utils.progressify(enumerate(self.L), desc="Budget constraints", total=len(self.L), enabled=progress):
                self.model.addLConstr(xs_l[i], GRB.LESS_EQUAL, bs_built_l[l],
                                      f"station-subset-only-mode-x_{i}-allowed-only-if-st-{self.L_map[l]}-built")
        else:
            # cumulative sums
            cs = self.model.addMVar(shape=(len(self.log), len(self.L_map)),
                                    vtype=GRB.INTEGER, name="c", lb=0, ub=self.max_allowed_budget)
            cs_l = cs.tolist()
            for j in range(len(self.L_map)):
                self.model.addLConstr(cs_l[0][j], GRB.EQUAL, bs_l[j])

            for i, l in utils.progressify(list(enumerate(self.L))[:-1], desc="Cumulative sums", total=len(self.L) - 1, enabled=progress):
                for j in range(l):
                    self.model.addLConstr(cs_l[i+1][j], GRB.EQUAL, cs_l[i][j])
                self.model.addLConstr(
                    cs_l[i+1][l], GRB.EQUAL, cs_l[i][l] - xs_l[i]*self.S[i])
                for j in range(l+1, len(self.L_map)):
                    self.model.addLConstr(cs_l[i+1][j], GRB.EQUAL, cs_l[i][j])

            # cumulative sum constraints
            if first_come_first_served:
                # signum variables
                ss = self.model.addMVar(
                    shape=len(self.log), vtype=GRB.BINARY, name="s")
                ss_l = ss.tolist()

                for gg in utils.progressify(self.A_groups, desc="Cumulative sum constraints", enabled=progress):
                    g = gg[::2]

                    for i in range(0, len(g)):
                        loc = self.L[g[i]]
                        self.model.addLConstr(
                            ss_l[g[i]], GRB.LESS_EQUAL, sum(self.xs[g[:i+1]].tolist()), f"signum-{g[i]}-<=-group-of-xs-{'-'.join(str(v) for v in g[:i+1].tolist())}")
                        # self.model.addLConstr(
                        # xs_l[g[i]], GRB.LESS_EQUAL, ss_l[g[i]], f"x-{g[i]}<=signum-{g[i]}")
                        self.model.addLConstr(
                            ss_l[g[i]], GRB.LESS_EQUAL, cs_l[g[i]][loc], f"signum-{g[i]}-<=-cumulative-sum-at-{g[i]}-at-location-{loc}-station-{self.L_map[loc]}")
                        self.model.addLConstr(
                            cs_l[g[i]][loc], GRB.LESS_EQUAL, self.max_allowed_budget * ss_l[g[i]], f"cumulative-sum-at-{g[i]}-at-location-{loc}-station-{self.L_map[loc]}-<=-{self.max_allowed_budget}*signum")
        return self

    def add_disallow_reattempts_constraints(self, progress=False):
        xs_l = self.xs.tolist()
        bs_built_l = self.bs_built.tolist()

        for g in utils.progressify(self.A_groups, desc="No reattempt constraints", enabled=progress):
            g = g[::2]

            for i1, i2 in zip(g[:-1], g[1:]):
                self.model.addLConstr(xs_l[i2], GRB.LESS_EQUAL, 1 - bs_built_l[self.L[i1]],
                                      f"no-reattempt-x{i2}-if-st-{self.L_map[self.L[i1]]}-of-x{i1}-built")

        self.reattempts_disallowed = True
        return self

    def clear_objective(self, priority: int):
        if priority in self.objective_constraints.keys():
            for cstr in self.objective_constraints[priority]:
                self.model.remove(cstr)
            self.objective_constraints[priority] = []

        return self

    def primary_optim_coverage_for_given_budget(self, budget: int, upper_bound: Optional[int] = None):
        assert not self.station_subset_only_mode, "Cannot optimize for given budget in station subset-only mode"
        PRIORITY = 1

        self.clear_objective(priority=PRIORITY)

        obj = -1 * sum(self.xs[np.where(self.st_mask)[0]].tolist())

        # -1 * to maximize
        self.model.setObjectiveN(obj, index=0, priority=PRIORITY)

        cstrs = []

        if upper_bound is not None:
            cstr = self.model.addLConstr(obj, GRB.GREATER_EQUAL, -1 * upper_bound,
                                         f"primary-optim-coverage-for-given-budget__known-upper-bound-{upper_bound}")
            cstrs.append(cstr)

        cstr = self.model.addLConstr(
            sum(self.bs.tolist()), GRB.LESS_EQUAL, budget, f"primary-optim-coverage-for-given-budget__sum-bs-<=-{budget}")
        cstrs.append(cstr)

        self.objective_constraints[PRIORITY] = cstrs

        return self

    def primary_optim_budget_for_full_coverage(self, progress=False):
        PRIORITY = 1
        self.clear_objective(priority=PRIORITY)

        self.model.setObjectiveN(
            sum(self.bs_built.tolist() if self.station_subset_only_mode else self.bs.tolist()), index=0, priority=PRIORITY)
        cstrs = []

        # attempt constraints
        for g in utils.progressify(self.A_groups, desc="Attempt constraints", enabled=progress):
            g = g[::2]
            # model.addLConstr(sum(xs[g].tolist()), GRB.LESS_EQUAL, 1, "m2_attempts")
            cstr = self.model.addLConstr(
                sum(self.xs[g].tolist()), GRB.EQUAL, 1, f"primary-optim-budget-for-full-coverage__xs-of-{'-'.join(str(v) for v in g.tolist())}==1")
            cstrs.append(cstr)

        cstrs.append(cstr)
        self.objective_constraints[PRIORITY] = cstrs

        return self

    def secondary_optim_distance_sum(self):
        PRIORITY = 0
        self.clear_objective(priority=PRIORITY)

        dists = self.log['station_distance'][self.st_mask].copy()
        dists /= dists.sum()

        obj = sum(
            x * d for x, d in zip(self.xs[np.where(self.st_mask)[0]].tolist(), dists.tolist()))

        self.model.setObjectiveN(obj, index=1, priority=PRIORITY)

        return self

    def secondary_optim_min_stations(self):
        assert not self.station_subset_only_mode, "Secondary min number of stations objective is redundant in station subset-only mode"
        PRIORITY = 0

        self.model.setObjectiveN(
            sum(self.bs_built.tolist()), index=1, priority=PRIORITY)

    def secondary_optim_closest_to_dist(self, budget_distribution):
        assert not self.station_subset_only_mode, "Cannot add secondary distance objective in station subset-only mode"
        bdist = [budget_distribution[l] for l in self.L_map]

        PRIORITY = 0
        self.clear_objective(priority=PRIORITY)
        cstrs = []

        bdiff = self.model.addMVar(
            shape=len(self.L_map), vtype=GRB.INTEGER, lb=-self.max_allowed_budget, ub=self.max_allowed_budget, name="budget-diff")
        bdiff_l = bdiff.tolist()
        cstrs.append(bdiff)

        bdiff_abs = self.model.addMVar(
            shape=len(self.L_map), vtype=GRB.INTEGER, lb=0, ub=self.max_allowed_budget, name="budget-diff-abs")
        bdiff_abs_l = bdiff_abs.tolist()
        cstrs.append(bdiff_abs)

        bs_l = self.bs.tolist()

        for i in range(len(self.L_map)):
            cstr = self.model.addConstr(
                bdiff_l[i] == bs_l[i] - bdist[i], f"secondary-optim-closest-to-dist__diff-of-{self.L_map[i]}")
            cstrs.append(cstr)

            cstr = self.model.addConstr(bdiff_abs_l[i] == gp.abs_(
                bdiff_l[i]), f"secondary-optim-closest-to-dist__diff-abs-of-{self.L_map[i]}")
            cstrs.append(cstr)

        self.model.setObjectiveN(sum(bdiff_abs_l), index=1, priority=PRIORITY)
        self.objective_constraints[PRIORITY] = cstrs

    def force_build_stations(self, budget_distribution, only_on_max_distance_violation=False):
        assert not self.station_subset_only_mode, "Cannot force build stations in station subset-only mode"
        for cstr in self.forced_built_stations_constraints:
            self.model.remove(cstr)
        self.forced_built_stations_constraints = []

        if budget_distribution is not None:
            sts = [not not budget_distribution[l] for l in self.L_map]

            bs_l = self.bs.tolist()
            if not only_on_max_distance_violation:
                for i in range(len(sts)):
                    if sts[i]:
                        cstr = self.model.addLConstr(
                            bs_l[i], GRB.GREATER_EQUAL, 1, f"station-{self.L_map[i]}-nonzero")
                        self.forced_built_stations_constraints.append(cstr)
            else:
                xs_l = self.xs.tolist()

                stations_first = self.log.query('attempt_no == 0 and status == 1')[
                    'station'].reindex(self.log.index).to_numpy()

                dist_mask = (self.log['station_distance']
                             > self.max_station_distance).to_numpy()
                st_mask = self.st_mask.to_numpy()

                for i in range(len(sts)):
                    if not sts[i]:
                        continue

                    ords = np.where(
                        (stations_first == self.L_map[i]) & dist_mask)[0]
                    for j in ords:
                        cstr = self.model.addLConstr(
                            bs_l[i], GRB.GREATER_EQUAL, xs_l[j], f"station-{self.L_map[i]}-nonzero-if-iloc-{j}")
                        self.forced_built_stations_constraints.append(cstr)

        return self

    def add_starter_solution(self, budget_distribution):
        assert not self.station_subset_only_mode, "Adding starter solution not yet supported in station subset-only mode"
        bs_start = [budget_distribution[l] for l in self.L_map]
        self.model.setAttr(GRB.Attr.Start, self.bs.tolist(), bs_start)

        if not self.reattempts_disallowed:
            xs_start = sim.simulate(sim.PreprocessedLog.from_dataframe(self.log),
                                    budget_distribution,
                                    max_station=self.log_original['station'].max(),
                                    max_station_distance=self.max_station_distance)
            xs_start = (xs_start.reindex(self.log.index, fill_value=-1)
                        == self.log['station']).tolist()

            self.model.setAttr(GRB.Attr.Start, self.xs.tolist(), xs_start)

        return self

    def check_solution_feasibility(self, budget_distribution):
        bs_start = [budget_distribution[l] for l in self.L_map]

        if self.station_subset_only_mode:
            bs_built_l = self.bs_built.tolist()
            for i, (b_var, b) in enumerate(zip(bs_built_l, bs_start)):
                built_flag = 1 if b >= 1 else 0
                self.model.addLConstr(
                    b_var, GRB.EQUAL, built_flag, f"b-built-{i}=={b}")
        else:
            bs_l = self.bs.tolist()
            for i, (b_var, b) in enumerate(zip(bs_l, bs_start)):
                self.model.addLConstr(b_var, GRB.EQUAL, b, f"b-{i}=={b}")

        if not self.reattempts_disallowed:
            xs_start = sim.simulate(sim.PreprocessedLog.from_dataframe(self.log),
                                    budget_distribution,
                                    max_station=self.log_original['station'].max(),
                                    max_station_distance=self.max_station_distance)
            xs_start = (xs_start.reindex(self.log.index, fill_value=-1)
                        == self.log['station']).tolist()

            xs_l = self.xs.tolist()

            for i, (x_var, x) in enumerate(zip(xs_l, xs_start)):
                self.model.addLConstr(x_var, GRB.EQUAL, x, f"x-{i}=={x}")

    def optimize(self, return_as_solution=True, return_simulation=False, compute_iis=False):
        self.model.setParam(GRB.Param.MIPFocus, 2)
        #self.model.setParam(GRB.Param.ConcurrentMIP, 2)
        self.model.setParam(GRB.Param.NodeMethod, 1)
        self.bs_built.setAttr(GRB.Attr.BranchPriority, 2)
        if self.bs is not None:
            self.bs.setAttr(GRB.Attr.BranchPriority, 1)
        self.model.update()

        print()
        self.model.optimize()
        print()

        if self.model.getAttr(GRB.Attr.Status) == GRB.INFEASIBLE:
            print("Infeasible!")

            if compute_iis:
                self.model.computeIIS()

                for v in self.model.getVars():
                    iis = dict(
                        lb=v.getAttr(GRB.Attr.IISLB),
                        ub=v.getAttr(GRB.Attr.IISUB),
                    )

                    if np.any(list(iis.values())):
                        print("Variable", v.getAttr(GRB.Attr.VarName), "LB =", v.getAttr(
                            GRB.Attr.LB), "UB =", v.getAttr(GRB.Attr.UB))

                for v in self.model.getConstrs():
                    iis = dict(
                        constr=v.getAttr(GRB.Attr.IISConstr)
                    )

                    if np.any(list(iis.values())):
                        print("Constraint", v.getAttr(GRB.Attr.ConstrName))

            return

        max_station = self.log_original['station'].max()

        xs_opt = self.xs.X > 0.5

        xs_log_full = self.log.iloc[xs_opt]

        xs_log = xs_log_full\
            .query('status == 1')['station']\
            .reindex(self.log_original.index.drop_duplicates(), fill_value=-1)

        if self.station_subset_only_mode:
            dist_opt = ListBudgetDist([0] * (max_station + 1))
            for l, b in enumerate(self.bs_built.X.astype(int)):
                if b >= 0.99:
                    dist_opt[self.L_map[l]] = min(
                        self.max_allowed_budget, 1)
        else:
            bs_opt = self.bs.X.astype(int)
            dist_opt = ListBudgetDist([0] * (max_station + 1))
            for l, b in enumerate(bs_opt):
                dist_opt[self.L_map[l]] = b

        if self.first_come_first_served and not self.station_subset_only_mode:
            sim_res = sim.simulate(
                sim.PreprocessedLog.from_dataframe(self.log_original),
                dist_opt, max_station=max_station,
                max_station_distance=self.max_station_distance)
            assert (sim_res == xs_log).all()

        if return_as_solution:
            result = optutils.Solution(
                budget=sum(dist_opt),
                distribution=dist_opt,
                objective=sim.get_satisfied_charging_requests(xs_log))
        else:
            result = dist_opt

        if return_simulation:
            return result, xs_log

        return result

    def close(self):
        self.model.dispose()


@dataclass(init=True, repr=True)
class ModelSpec:
    log: Union[pd.DataFrame, trf.TrafficSpec]
    budget_upper_limit: int
    max_station_distance: float

    opt_coverage_total_budget: Optional[int] = None
    opt_budget_for_full_coverage: bool = False

    secondary_opt_distance_sum: bool = False
    secondary_opt_min_stations: bool = False

    secondary_optim_closest_to_distribution: Optional[Union[List[int],
                                                            BudgetDist]] = None

    force_build_stations_distribution: Optional[Union[List[int],
                                                      BudgetDist]] = None
    force_build_stations_on_max_distance_violation_only: bool = False

    station_subset_only_mode: bool = False

    starter_solution_distribution: Optional[Union[List[int],
                                                  BudgetDist]] = None
    test_starter_solution: bool = False

    common_constraints: bool = True
    first_come_first_served: bool = True

    disallow_reattempts: bool = False

    debug: bool = False
    progress: bool = False

    def __post_init__(self):
        assert (
            self.opt_coverage_total_budget is None) == self.opt_budget_for_full_coverage
        assert self.opt_budget_for_full_coverage is None or self.budget_upper_limit >= self.opt_budget_for_full_coverage
        assert self.test_starter_solution == False or self.starter_solution_distribution != None

        nsecondaries = sum([int(b) for b in [
            self.secondary_opt_distance_sum,
            self.secondary_opt_min_stations,
            self.secondary_optim_closest_to_distribution is not None,
        ]])

        assert nsecondaries <= 1

        assert not self.station_subset_only_mode or (
            (self.test_starter_solution or self.starter_solution_distribution == None) and
            self.opt_budget_for_full_coverage and
            (self.secondary_opt_distance_sum or nsecondaries == 0) and
            self.force_build_stations_distribution == None)


def build_model(spec: ModelSpec,
                traffic_full: Optional[pd.DataFrame] = None,
                charging_stations: Optional[pd.DataFrame] = None,
                station_distances_mtx: Optional[List[List[float]]] = None) -> _ModelBuilder:
    mb = _ModelBuilder()
    mb.enter()
    mb.set_debug(spec.debug)

    max_allowed_budget = spec.budget_upper_limit
    if spec.starter_solution_distribution is not None:
        max_allowed_budget = min(spec.budget_upper_limit, sum(
            spec.starter_solution_distribution))

    log: pd.DataFrame
    if isinstance(spec.log, trf.TrafficSpec):
        if traffic_full is None:
            raise Exception(
                "traffic_full must be specified if spec.log is an instance of TrafficSpec")
        elif charging_stations is None:
            raise Exception(
                "charging_stations must be specified if spec.log is an instance of TrafficSpec")
        elif station_distances_mtx is None:
            raise Exception(
                "station_distances_mtx must be specified if spec.log is an instance of TrafficSpec")
        else:
            log = sim.traffic_to_log(trf.build_traffic(
                spec.log, traffic_full, charging_stations, station_distances_mtx))
    elif isinstance(spec.log, pd.DataFrame):
        log = spec.log
    else:
        raise Exception(f"Unexpected value type {type(spec.log)} of spec.log")

    mb.init_log(log, max_allowed_budget=max_allowed_budget)
    mb.init_settings(station_subset_only_mode=spec.station_subset_only_mode)
    mb.limit_station_distance(spec.max_station_distance)
    mb.add_variables()
    if spec.common_constraints:
        mb.add_common_constraints(
            first_come_first_served=spec.first_come_first_served, progress=spec.progress)

    if spec.disallow_reattempts:
        mb.add_disallow_reattempts_constraints(progress=spec.progress)

    if spec.opt_coverage_total_budget is not None:
        mb.primary_optim_coverage_for_given_budget(
            spec.opt_coverage_total_budget)
    elif spec.opt_budget_for_full_coverage:
        mb.primary_optim_budget_for_full_coverage(progress=spec.progress)

    if spec.secondary_opt_distance_sum:
        mb.secondary_optim_distance_sum()
    elif spec.secondary_opt_min_stations:
        mb.secondary_optim_min_stations()
    elif spec.secondary_optim_closest_to_distribution is not None:
        mb.secondary_optim_closest_to_dist(
            list(spec.secondary_optim_closest_to_distribution))

    if spec.force_build_stations_distribution is not None:
        mb.force_build_stations(list(spec.force_build_stations_distribution),
                                spec.force_build_stations_on_max_distance_violation_only)

    if spec.test_starter_solution:
        mb.check_solution_feasibility(spec.starter_solution_distribution)
    elif spec.starter_solution_distribution is not None:
        mb.add_starter_solution(list(spec.starter_solution_distribution))

    return mb
