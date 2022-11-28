"""
Script to generate training data of the Exam Timetabling (ET) problem
"""
import os
from datetime import datetime

import pandas as pd

from instance import INSTANCE_SPACE, get_ET_instance
from solver import CplexSolver


def generate_training_data(configuration_space, parent_folder : str):
    """
    Generates a training dataset and stores the results in a folder

    Inputs
    -------
    
    configuration_space : list[dict]
        list of dictionaries where each dictionary contains solver configuration 
        parameters 

    num_instances : int
        number of instances to solver for each solver configuration

    parent_folder : str
        the name of the folder to save the results

    The results will be store inside the parent folder with this structure:

    Folder structure:

    parent_folder
        generated_codename
        |   config_1.txt
        |   config_2.txt
        ...
        |   config_N.txt
        |   config_info.csv

    Where each config_i.txt includes:

        col1: x variable value
        col2: y variable value
        col3: run time    
        col4: objective value
        col5: whether a feasible solution was found
        col6: weather the optimal solution was found

    for each instance solved (number rows = number instances solved)


    """
    folder_codename = datetime.now().strftime("%y_%m_%d_%H_%M")

    # create an inventory of the solver configuration space
    df = pd.DataFrame(columns=configuration_space[0].keys())
    for i, config_parameters in enumerate(configuration_space):
        df.loc[i] = config_parameters

    path = f"{parent_folder}/{folder_codename}"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    df.to_csv(f"{path}/config_info.csv")

    # loop through configuration space
    for i, configuration_parameters in enumerate(configuration_space):
        # solver num_instances for each configuration
        for instance_num in range(len(INSTANCE_SPACE)):
            solver = CplexSolver()
            solver.initialize_solver(configuration_parameters)
            save_filepath = f"{path}/config_{i}/instance_{instance_num}.txt"
            instance = get_ET_instance(instance_num)
            solver.solve(instance, save_filepath=save_filepath)

    print("Finished generating the training data")


def parse_solutions(folder_name):
    """
    Parses a folder and generates a training set of
    configuration parameters and solution values

    Folder structure:

        folder_name
        |   config_1.txt
        |   config_2.txt
        ...
        |   config_N.txt
        |   config_info.csv

    Where each config_i.txt includes:

        x variable value
        y variable value
        run time    
        objective value
        whether a feasible solution was found
        weather the optimal solution was found

    for each instance solved (number rows = number instances solved)

    """
    pass


if __name__ == "__main__":
    timelimit = 86400

    configuration_space = [
        {"timelimit" : timelimit, "lpmethod" : 2},
        {"timelimit" : timelimit, "lpmethod" : 0},
        {"timelimit" : timelimit, "lpmethod" : 1},
    ]


    generate_training_data(
        configuration_space=configuration_space, 
        parent_folder="cplex_results")
