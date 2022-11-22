import os

import cplex
import docplex
import numpy as np
from docplex.mp.model import Model
from docplex.mp.progress import TextProgressListener
from docplex.mp.solution import SolveSolution
from tqdm import tqdm

from instance import ET_Instance


class CplexSolver:
    def __init__(self):
        self.optimizer : Model

    def initialize_solver(self, configuration_parameters):
        """
        Initializes a cplex solver with the given configuration parameters
        """
        timelimit = configuration_parameters["timelimit"]

        self.optimizer = Model(name='solver')
        self.optimizer.parameters.timelimit = timelimit

        return

    def solve(self, problem_instance, save_filepath : str = ''):
        """
        Solves a problem instance
        """
        E = problem_instance.exam_set
        S = problem_instance.student_set
        R = problem_instance.room_set
        T = problem_instance.datetime_slot_set
        Cp = problem_instance.room_capacity_set
        He_s = problem_instance.courses_enrollments_set
        ratio_inv_students = problem_instance.ratio_inv_students
        sumHe_s = np.sum(He_s, axis=1)

        x = self.optimizer.binary_var_matrix(len(E), len(T), name="X_e,t") # whether we use timeslot t for exam e
        y = self.optimizer.binary_var_matrix(len(E), len(R), name="Y_e,r") # whether we use room r for exam e
        c1 = self.optimizer.add_constraints((sum(x[e, t] for t in range(len(T))) >= 1 for e in range(len(E))), names='c1') 
        c2 = self.optimizer.add_constraints((sum(y[e, r] for r in range(len(R))) == 1 for e in range(len(E))), names='c2') 

        # c3 modified constraint 
        for s in tqdm(range(len(S))):
            for t in range(len(T)):
                cond = sum(x[e,t] * He_s[e,s] for e in range(len(E)))
                if type(cond) != int:
                    self.optimizer.add_constraint(cond <= 1)

        # c4 modified constraint
        for r in tqdm(range(len(R))):
            for t in range(len(T)):
                cond = sum((x[e,t]*y[e,r]) * sumHe_s[e] for e in range(len(E)))
                if type(cond) != int:
                    self.optimizer.add_constraint(cond <= Cp[r])

        # objective function
        obj_fun =  sum(sum(y[e,r] * sumHe_s[e] for e in range(len(E))) for r in range(len(R)))
        self.optimizer.set_objective('min', obj_fun)
        self.optimizer.print_information()
        self.optimizer.add_progress_listener(TextProgressListener())

        # solve
        sol = self.optimizer.solve(clean_before_solve=True)
        
        # process the solution
        if sol:
            print("Found a solution \n")            
            solve_time = self.optimizer.solve_details.time
            objective_value = self.optimizer.objective_value
            
        else:
            print("Could not find a solution")
            solve_time = np.nan
            objective_value = np.nan

        # write to file
        if len(save_filepath):
            path = save_filepath.split("/instance")[0]
            
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            
            for v in self.optimizer.iter_binary_vars():
                with open(save_filepath, 'a') as f:
                    if sol:
                        f.write(f"{v} = {v.solution_value} \n")
                    else:
                        f.write(f"{v} = {np.nan} \n")
        
            with open(save_filepath, 'a') as f:
                f.write(f"rt = {solve_time} \n")
                f.write(f"obj = {objective_value} \n")

        return