import numpy as np
from ConfigSpace import ConfigurationSpace
from sklearn.ensemble import RandomForestRegressor
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.acquisition.function import PriorAcquisitionFunction

from instance import INSTANCE_SPACE, get_ET_instance
from solver import CplexSolver


def minimize_mip_runtime(config, seed: int = 0):
    run_times_list = []

    mip_static_config =  {"timelimit" : 86400}

    print(config)

    for instance_num in range(len(INSTANCE_SPACE)):
        mip_solver = CplexSolver()
        mip_solver.initialize_solver({**config, **mip_static_config})
        instance = get_ET_instance(instance_num)
        run_time = mip_solver.solve(instance)
        run_times_list.append(run_time)

    print(np.mean(run_times_list))

    return np.mean(run_times_list)


if __name__ == "__main__":
    configspace = ConfigurationSpace({
        "barrier_algorithm": [1, 2, 3],
        "lpmethod": [0, 1, 2, 3],
        "mip_s_bbinterval": [0, 1, 7],  #default 7
        "mip_branching_direction": [-1, 0, 1], #default 0
        "backtracking_tolerance": [np.random.uniform(0.000000001,0.999999999)], #default 0.99
        "mip_cliques_switch":[-1,0,1,2,3], #default 0
        "coefficient_reduction_setting": [-1,0,1,2,3], #default -1
        "mip_covers_switch": [-1,0,1,2,3], #default -1
        "number_of_cutting_plane_passes": [-1, 0], #any postive int, default 0
        "cut_factor_row_multiplier_limit": [-1, 0], #any postive float, 
        "mip_dive_strategy": [0, 1, 2, 3], #default 0
        "dual_simplex_pricing_algorithm": [0, 1, 2, 3, 4, 5], #default 0
        "mip_subproblem_algorithm": [1, 2, 3, 4, 5], #default 0
        "mip_node_selection_strategy": [0, 1, 2, 3], #default 1
        "mip_variable_selection_strategy": [-1, 0, 1, 2, 3, 4] #default 0
    })
    
    scenario = Scenario(configspace, n_trials=20)

    smac = ACFacade(scenario, minimize_mip_runtime)
    best_found_config = smac.optimize()