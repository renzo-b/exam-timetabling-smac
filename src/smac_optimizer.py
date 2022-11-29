import os
from datetime import datetime

import numpy as np
import pandas as pd
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.ensemble import RandomForestRegressor
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.acquisition.function import PriorAcquisitionFunction
from smac.initial_design.default_design import DefaultInitialDesign

from instance import INSTANCE_SPACE, get_ET_instance
from solver import CplexSolver


def minimize_mip(config, seed: int = 0):
    folder_codename = datetime.now().strftime("%y_%m_%d_%H_%M")
    path = f"cplex_results/{folder_codename}"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    df = pd.DataFrame(columns=config.keys())
    for i, config_parameters in enumerate([config]):
        df.loc[i] = config_parameters
    df.to_csv(f"{path}/config_info.csv")

    mip_static_config =  {"timelimit" : 1800, "threads" : 2}
    objective_value_list = []

    print('Trying configuration: ')
    print(config)

    for instance_num in range(len(INSTANCE_SPACE)):
        mip_solver = CplexSolver()
        mip_solver.initialize_solver({**config, **mip_static_config})
        save_filepath = f"{path}/instance_{instance_num}.txt"
        instance = get_ET_instance(instance_num)
        print(f"you're about to solve for instance num: {instance_num} out of {len(INSTANCE_SPACE)}")
        objective_value = mip_solver.solve(instance, save_filepath=save_filepath, verbose=True)
        objective_value_list.append(objective_value)

    return np.mean(objective_value_list)


if __name__ == "__main__":
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(Categorical("lpmethod", [0, 1, 2, 3], default=0))
    configspace.add_hyperparameter(Categorical("mip_s_bbinterval", [0, 1, 7], default=7))
    configspace.add_hyperparameter(Categorical("mip_branching_direction", [-1, 0, 1], default=0))
    configspace.add_hyperparameter(Float("backtracking_tolerance", [0.000000001,0.999999999], default=0.99))
    configspace.add_hyperparameter(Categorical("mip_cliques_switch",[-1,0,1,2,3], default=0))
    configspace.add_hyperparameter(Categorical("coefficient_reduction_setting", [-1,0,1,2,3], default=-1))
    configspace.add_hyperparameter(Categorical("mip_covers_switch", [-1,0,1,2,3], default=0))
    configspace.add_hyperparameter(Float("number_of_cutting_plane_passes", [-1, 0], default=0))
    configspace.add_hyperparameter(Integer("cut_factor_row_multiplier_limit", [-1, 10], default=-1)) #any postive float, 
    configspace.add_hyperparameter(Categorical("mip_dive_strategy", [0, 1, 2, 3], default=0))
    configspace.add_hyperparameter(Categorical("dual_simplex_pricing_algorithm", [0, 1, 2, 3, 4, 5], default=0)) 
    configspace.add_hyperparameter(Categorical("mip_subproblem_algorithm", [0, 1, 2, 3, 4, 5], default=0))
    configspace.add_hyperparameter(Categorical("mip_node_selection_strategy", [0, 1, 2, 3], default=1))
    configspace.add_hyperparameter(Categorical("mip_variable_selection_strategy", [-1, 0, 1, 2, 3, 4], default=0))
    
    scenario = Scenario(configspace, name='cplex_run_time_1', n_trials=30)

    default_design = DefaultInitialDesign(scenario)

    smac = ACFacade(
        scenario=scenario, 
        target_function=minimize_mip, 
        initial_design=default_design,
        logging_level=1,
        )
    best_found_config = smac.optimize()

    print("BEST CONFIG:")
    print(best_found_config)