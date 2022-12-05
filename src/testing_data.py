import os
from datetime import datetime

import numpy as np
import pandas as pd

from instance import TEST_INSTANCE_SPACE, TEST_SEMESTERS, get_ET_instance
from solver import CplexSolver

SMAC_RUN_NAME = "test_run"
CPLEX_TIME_LIMIT = 1200  # seconds
MIP_GAP = 0.01  # 1 %
random_seed = 10


def config_test(config, seed: int = 1):
    folder_codename = SMAC_RUN_NAME + "_" + datetime.now().strftime("%y_%m_%d_%H_%M")
    path = f"cplex_results/{folder_codename}"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    mip_static_config = {
        "timelimit": CPLEX_TIME_LIMIT,
        "mipgap": MIP_GAP,
        "random_seed": random_seed,
    }
    objective_value_list = []

    print("\n Trying configuration: ")
    print(config)

    for instance_num in range(len(TEST_INSTANCE_SPACE)):
        mip_solver = CplexSolver()
        mip_solver.initialize_solver({**config, **mip_static_config})
        save_filepath = f"{path}/instance_{instance_num}.txt"
        instance = get_ET_instance(instance_num)
        print(
            f"you're about to solve for instance num: {instance_num} out of {len(TEST_INSTANCE_SPACE)}"
        )
        runtime, _, df_schedule = mip_solver.solve(
            instance, save_filepath=save_filepath, verbose=True
        )
        objective_value_list.append(runtime)
        df_schedule.to_csv(f"{path}/DF_Schedual_{instance_num}.csv")

    return np.mean(objective_value_list)


if __name__ == "__main__":
    # replace with the folder that contains the config you want to test
    folder = "cplex_results/SMAC_OPT_COST_001_22_12_04_21_10_34"
    config = pd.read_csv(f"{folder}/config_info.csv").to_dict(orient="records")

    del config[0]["Unnamed: 0"]

    config = config[0]

    config_test(config)
