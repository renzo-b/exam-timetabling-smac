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
        "lpmethod": [0, 1, 2, 3],
        "bbinterval": [0, 1, 7],
        "mip_branching_direction": [-1, 0, 1],
    })
    
    scenario = Scenario(configspace, n_trials=20)

    smac = ACFacade(scenario, minimize_mip_runtime)
    best_found_config = smac.optimize()