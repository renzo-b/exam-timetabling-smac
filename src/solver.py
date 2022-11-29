import os

import numpy as np
from docplex.mp.model import Model
from docplex.mp.progress import TextProgressListener
from tqdm import tqdm


class CplexSolver:
    def __init__(self):
        self.optimizer: Model

    def initialize_solver(self, configuration_parameters):
        """
        Initializes a cplex solver with the given configuration parameters
        """
        timelimit = configuration_parameters["timelimit"]
        lpmethod = configuration_parameters["lpmethod"]
        mip_s_bbinterval = configuration_parameters["mip_s_bbinterval"]
        mip_branching_direction = configuration_parameters["mip_branching_direction"]
        backtracking_tolerance = configuration_parameters["backtracking_tolerance"]
        mip_cliques_switch = configuration_parameters["mip_cliques_switch"]
        coefficient_reduction_setting = configuration_parameters["coefficient_reduction_setting"]
        mip_covers_switch = configuration_parameters["mip_covers_switch"]
        number_of_cutting_plane_passes = configuration_parameters["number_of_cutting_plane_passes"]
        cut_factor_row_multiplier_limit = configuration_parameters[
            "cut_factor_row_multiplier_limit"]
        mip_dive_strategy = configuration_parameters["mip_dive_strategy"]
        dual_simplex_pricing_algorithm = configuration_parameters["dual_simplex_pricing_algorithm"]
        mip_subproblem_algorithm = configuration_parameters["mip_subproblem_algorithm"]
        mip_node_selection_strategy = configuration_parameters["mip_node_selection_strategy"]
        mip_variable_selection_strategy = configuration_parameters[
            "mip_variable_selection_strategy"]

        self.optimizer = Model(name='solver')
        self.optimizer.parameters.timelimit = timelimit
        self.optimizer.parameters.lpmethod = lpmethod
        self.optimizer.parameters.mip.strategy.bbinterval = mip_s_bbinterval
        self.optimizer.parameters.mip.strategy.branch = mip_branching_direction
        self.optimizer.parameters.mip.strategy.backtrack = backtracking_tolerance
        self.optimizer.parameters.mip.cuts.cliques = mip_cliques_switch
        self.optimizer.parameters.preprocessing.coeffreduce = coefficient_reduction_setting
        self.optimizer.parameters.mip.cuts.covers = mip_covers_switch
        self.optimizer.parameters.mip.limits.cutpasses = number_of_cutting_plane_passes
        self.optimizer.parameters.mip.limits.cutsfactor = cut_factor_row_multiplier_limit
        self.optimizer.parameters.mip.strategy.dive = mip_dive_strategy
        self.optimizer.parameters.simplex.dgradient = dual_simplex_pricing_algorithm
        self.optimizer.parameters.mip.strategy.subalgorithm = mip_subproblem_algorithm
        self.optimizer.parameters.mip.strategy.nodeselect = mip_node_selection_strategy
        self.optimizer.parameters.mip.strategy.variableselect = mip_variable_selection_strategy

        return

    def add_variables(self, E, T, R):
        """
        """
        x = self.optimizer.binary_var_matrix(len(E), len(
            T), name="X_e,t")  # whether we use timeslot t for exam e
        y = self.optimizer.binary_var_matrix(len(E), len(
            R), name="Y_e,r")  # whether we use room r for exam e
        x_etr = self.optimizer.binary_var_cube(
            len(E), len(T), len(R), name='xetr')

        return x, y, x_etr

    def add_constraints(self, E, S, T, R, Cp, He_s, sumHe_s, x, y, x_etr):
        print("Loading constraints")
        self.optimizer.add_constraints(
            (sum(x[e, t] for t in range(len(T))) == 1
                for e in range(len(E))), names='c1')
        self.optimizer.add_constraints(
            (sum(y[e, r] for r in range(len(R))) >= 1
                for e in range(len(E))), names='c2')
        self.optimizer.add_constraints(
            (sum(x_etr[e, t, r] for r in range(len(R))) == x[e, t]
                for e in tqdm(range(len(E))) for t in range(len(T))))
        self.optimizer.add_constraints(
            (sum(x_etr[e, t, r] for t in range(len(T))) == y[e, r]
                for e in tqdm(range(len(E))) for r in range(len(R))))
        self.optimizer.add_constraints(
            (sum(x_etr[e, t, r] * sumHe_s[e] for e in range(len(E))) <= Cp[r]
                for r in tqdm(range(len(R))) for t in range(len(T))))
        # c6
        for s in tqdm(range(len(S))):
            for t in range(len(T)):
                cond = sum(x[e, t] * He_s[e, s] for e in range(len(E)))
                if type(cond) != int:
                    self.optimizer.add_constraint(cond <= 1)

    def add_situational_constraints(self, E, R, x, x_etr, room_availability, prof_availability):
        for idx in np.argwhere(room_availability == 1):
            r = idx[0]
            t = idx[1]

            self.optimizer.add_constraints(
                (x_etr[e, t, r] == 0) for e in range(len(E)))

        for idx in np.argwhere(prof_availability == 1):
            e = idx[0]
            t = idx[1]

            self.optimizer.add_constraint(x[e, t] == 0)
            self.optimizer.add_constraints(
                (x_etr[e, t, r] == 0) for r in range(len(R)))

    def add_objective_function(self, y, E, R, sumHe_s, ratio_of_Inv):
        up = (sum(1 * sumHe_s[e] * ratio_of_Inv for e in range(len(E)))
              for r in range(len(R)))
        upper_bound = 0
        for i in up:
            upper_bound += np.ceil(i)

        ceil_obj = []
        sum_sum = []

        for r in range(len(R)):
            ceil_obj.append(self.optimizer.integer_var(lb=0, ub=upper_bound))
            sum_sum.append(sum(y[e, r] * sumHe_s[e] *
                               ratio_of_Inv for e in range(len(E))))
            self.optimizer.add_constraint(ceil_obj[r] >= sum_sum[r])

        obj_fun = sum(ceil_obj[r] for r in range(len(R)))

        # Optimizer Info
        self.optimizer.set_objective('min', obj_fun)
        self.optimizer.print_information()

    def solve(self, problem_instance, save_filepath: str = '', verbose=False):
        """
        Solves a problem instance
        """
        # Instance sets
        E = problem_instance.exam_set
        S = problem_instance.student_set
        R = problem_instance.room_set
        T = problem_instance.datetime_slot_set
        Cp = problem_instance.room_capacity_set
        He_s = problem_instance.courses_enrollments_set
        ratio_of_Inv = problem_instance.ratio_inv_students
        sumHe_s = np.sum(He_s, axis=1)
        room_availability = problem_instance.room_availability
        prof_availability = problem_instance.prof_availability

        # Variables
        x, y, x_etr = self.add_variables(E, T, R)

        # Constraints
        self.add_constraints(E, S, T, R, Cp, He_s, sumHe_s, x, y, x_etr)

        # Optional Constraints
        self.add_situational_constraints(
            E, R, x, x_etr, room_availability, prof_availability)

        # Objective Function
        self.add_objective_function(y, E, R, sumHe_s, ratio_of_Inv)

        # Solve
        if verbose:
            self.optimizer.add_progress_listener(TextProgressListener())
            log_output = True
        else:
            log_output = False
        sol = self.optimizer.solve(
            log_output=log_output, clean_before_solve=True)

        # process the solution
        if sol:
            print("Found a solution \n")
            solve_time = self.optimizer.solve_details.time
            objective_value = self.optimizer.objective_value

        else:
            print("Could not find a solution")
            solve_time = 1e6
            objective_value = 1e6

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

        self.optimizer.clear()

        return objective_value
