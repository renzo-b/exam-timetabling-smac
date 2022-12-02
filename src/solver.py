import os
from math import ceil, floor

import numpy as np
import pandas as pd
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
        mipgap = configuration_parameters["mipgap"]
        lpmethod = configuration_parameters["lpmethod"]
        mip_s_bbinterval = configuration_parameters["mip_s_bbinterval"]
        mip_branching_direction = configuration_parameters["mip_branching_direction"]
        backtracking_tolerance = configuration_parameters["backtracking_tolerance"]
        mip_cliques_switch = configuration_parameters["mip_cliques_switch"]
        coefficient_reduction_setting = configuration_parameters[
            "coefficient_reduction_setting"
        ]
        mip_covers_switch = configuration_parameters["mip_covers_switch"]
        number_of_cutting_plane_passes = configuration_parameters[
            "number_of_cutting_plane_passes"
        ]
        cut_factor_row_multiplier_limit = configuration_parameters[
            "cut_factor_row_multiplier_limit"
        ]
        mip_dive_strategy = configuration_parameters["mip_dive_strategy"]
        dual_simplex_pricing_algorithm = configuration_parameters[
            "dual_simplex_pricing_algorithm"
        ]
        mip_subproblem_algorithm = configuration_parameters["mip_subproblem_algorithm"]
        mip_node_selection_strategy = configuration_parameters[
            "mip_node_selection_strategy"
        ]
        mip_variable_selection_strategy = configuration_parameters[
            "mip_variable_selection_strategy"
        ]

        self.optimizer = Model(name="solver")
        self.optimizer.parameters.timelimit = timelimit
        self.optimizer.parameters.mip.tolerances.mipgap = mipgap
        self.optimizer.parameters.lpmethod = lpmethod
        self.optimizer.parameters.mip.strategy.bbinterval = mip_s_bbinterval
        self.optimizer.parameters.mip.strategy.branch = mip_branching_direction
        self.optimizer.parameters.mip.strategy.backtrack = backtracking_tolerance
        self.optimizer.parameters.mip.cuts.cliques = mip_cliques_switch
        self.optimizer.parameters.preprocessing.coeffreduce = (
            coefficient_reduction_setting
        )
        self.optimizer.parameters.mip.cuts.covers = mip_covers_switch
        self.optimizer.parameters.mip.limits.cutpasses = number_of_cutting_plane_passes
        self.optimizer.parameters.mip.limits.cutsfactor = (
            cut_factor_row_multiplier_limit
        )
        self.optimizer.parameters.mip.strategy.dive = mip_dive_strategy
        self.optimizer.parameters.simplex.dgradient = dual_simplex_pricing_algorithm
        self.optimizer.parameters.mip.strategy.subalgorithm = mip_subproblem_algorithm
        self.optimizer.parameters.mip.strategy.nodeselect = mip_node_selection_strategy
        self.optimizer.parameters.mip.strategy.variableselect = (
            mip_variable_selection_strategy
        )

        return

    def add_variables(self, E, T, R, S):
        """ """
        x = self.optimizer.binary_var_matrix(
            len(E), len(T), name="X_e,t"
        )  # whether we use timeslot t for exam e
        y = self.optimizer.binary_var_matrix(
            len(E), len(R), name="Y_e,r"
        )  # whether we use room r for exam e
        x_etr = self.optimizer.binary_var_cube(
            len(E), len(T), len(R), name="xetr")

        x_st = self.optimizer.binary_var_matrix(
            len(S), len(T), name="x_st"
        )  # whether student s in sitting in an exam at time T

        return x, y, x_etr, x_st

    def add_constraints(self, E, S, T, R, Cp, He_s, sumHe_s, x, y, x_etr, x_st):
        print("Loading constraints")
        self.optimizer.add_constraints(
            (sum(x[e, t] for t in range(len(T))) == 1 for e in range(len(E))),
            names="c1",
        )
        self.optimizer.add_constraints(
            (sum(y[e, r] for r in range(len(R))) >= 1 for e in range(len(E))),
            names="c2",
        )
        self.optimizer.add_constraints(
            (
                sum(x_etr[e, t, r] for r in range(len(R))) == x[e, t]
                for e in tqdm(range(len(E)))
                for t in range(len(T))
            )
        )
        self.optimizer.add_constraints(
            (
                sum(x_etr[e, t, r] for t in range(len(T))) == y[e, r]
                for e in tqdm(range(len(E)))
                for r in range(len(R))
            )
        )
        self.optimizer.add_constraints(
            (
                sum(x_etr[e, t, r] * sumHe_s[e]
                    for e in range(len(E))) <= Cp[r]
                for r in tqdm(range(len(R)))
                for t in range(len(T))
            )
        )
        # c6
        for s in tqdm(range(len(S))):
            for t in range(len(T)):
                cond = sum(x[e, t] * He_s[e, s] for e in range(len(E)))

                self.optimizer.add_constraint(cond == x_st[s, t])

                if type(cond) != int:
                    self.optimizer.add_constraint(cond <= 1)

        # C8 only one exam per day for each student
        for s in range(len(S)):
            k = 0
            for i in range(ceil(len(T) / 3)):
                if i == ceil(len(T) / 3) - 1:
                    sum_xt = 0
                    for j in range(len(T) % 3):
                        sum_xt += x_st[s, k + j]
                    self.optimizer.add_constraint(sum_xt <= 1)
                else:
                    self.optimizer.add_constraint(
                        (x_st[s, k] + x_st[s, k + 1] + x_st[s, k + 2]) <= 1
                    )
                k += 3

    def add_situational_constraints(
        self, E, R, x, x_etr, room_availability, prof_availability
    ):
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
        up = (
            sum(1 * sumHe_s[e] * ratio_of_Inv for e in range(len(E)))
            for r in range(len(R))
        )
        upper_bound = 0
        for i in up:
            upper_bound += np.ceil(i)

        ceil_obj = []
        sum_sum = []

        for r in range(len(R)):
            ceil_obj.append(self.optimizer.integer_var(lb=0, ub=upper_bound))
            sum_sum.append(
                sum(y[e, r] * sumHe_s[e] * ratio_of_Inv for e in range(len(E)))
            )
            self.optimizer.add_constraint(ceil_obj[r] >= sum_sum[r])

        obj_fun = sum(ceil_obj[r] for r in range(len(R)))

        # Optimizer Info
        self.optimizer.set_objective("min", obj_fun)
        self.optimizer.print_information()

    def solve(self, problem_instance, save_filepath: str = "", verbose=False):
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
        x, y, x_etr, x_st = self.add_variables(E, T, R, S)

        # Constraints
        self.add_constraints(E, S, T, R, Cp, He_s, sumHe_s, x, y, x_etr, x_st)

        # Optional Constraints
        self.add_situational_constraints(
            E, R, x, x_etr, room_availability, prof_availability
        )

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

        print(self.optimizer.solve_details.status, ' status code: ',
              self.optimizer.solve_details.status_code)

        # process the solution
        if self.optimizer.solve_details.status_code == 101:  # success
            print('processing success')
            schedule, df_x, df_y = self.process_solution(sol, x, y, E, T, R)
            enrolment_df = create_enrolment_df(He_s, S, E)
            df_schedule = (schedule.merge(enrolment_df, on="EXAM", how="left")).drop(
                ["exam_x", "value_x", "exam_y", "room", "value_y"], axis=1
            )
            solve_time = self.optimizer.solve_details.time
            objective_value = self.optimizer.objective_value
            status = "solution"

        if self.optimizer.solve_details.status_code == 102:  # success
            print('processing success w/ tolerance')
            schedule, df_x, df_y = self.process_solution(sol, x, y, E, T, R)
            enrolment_df = create_enrolment_df(He_s, S, E)
            df_schedule = (schedule.merge(enrolment_df, on="EXAM", how="left")).drop(
                ["exam_x", "value_x", "exam_y", "room", "value_y"], axis=1
            )
            solve_time = self.optimizer.solve_details.time
            objective_value = self.optimizer.solve_details.best_bound
            status = "bound"

        elif self.optimizer.solve_details.status_code == 108:  # time limit
            print('processing timelimit')
            df_schedule = pd.DataFrame({"F": ["Failed :("]})
            solve_time = self.optimizer.solve_details.time
            objective_value = self.optimizer.solve_details.best_bound
            status = "timeout"

        elif self.optimizer.solve_details.status_code == 103:  # infeasible problem
            print('processing infeasible')
            df_schedule = pd.DataFrame({"F": ["Failed :("]})
            solve_time = self.optimizer.solve_details.time
            objective_value = 0
            status = "infeasible"

        else:
            solve_time = 1e6
            objective_value = 1e6
            df_schedule = pd.DataFrame({"F": ["Failed :("]})
            status = "unknown"

        # write to file
        if len(save_filepath):
            path = save_filepath.split("/instance")[0]

            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            for v in self.optimizer.iter_binary_vars():
                with open(save_filepath, "a") as f:
                    # if sol:
                    f.write(f"{v} = {v.solution_value} \n")
                    # else:
                    #     f.write(f"{v} = {np.nan} \n")

            with open(save_filepath, "a") as f:
                f.write(f"rt = {solve_time} \n")
                f.write(f"obj = {objective_value} \n")
                f.write(f"status = {status} \n")

        self.optimizer.clear()

        return solve_time, objective_value, df_schedule

    def process_solution(self, sol, x, y, E, T, R):
        """
        Takes a cplex solution and produces a exam schedule

        Parameters
        ----------
        sol : SolveSolution
            solution from the solver

        Returns
        -------
        final_schedule : pd.DataFrame
            The schedule formatted in readable format for an exam organizer

        df_x : pd.DataFrame
            The results for variable x

        df_y : pd.DataFrame
            The results for variable y
        """
        # extract solutions as df
        df_x = sol.get_value_df(x).rename(
            columns={"key_1": "exam", "key_2": "timeslot"}
        )
        df_y = sol.get_value_df(y).rename(
            columns={"key_1": "exam", "key_2": "room"})

        # Add rows with the names of courses and timelots
        exam_col = [E[i] for i in range(len(E)) for j in range(len(T))]
        time_col = [T[j] for i in range(len(E)) for j in range(len(T))]
        df_x["EXAM"] = exam_col
        df_x["TIMESLOT"] = time_col

        # Add rows with the names of courses and rooms
        exam_col = [E[i] for i in range(len(E)) for j in range(len(R))]
        room_col = [R[j] for i in range(len(E)) for j in range(len(R))]
        df_y["EXAM"] = exam_col
        df_y["ROOM"] = room_col

        # Produce the final schedule
        final_schedule = df_x[df_x["value"] == 1].merge(
            df_y[df_y["value"] == 1], on="EXAM", how="left"
        )
        final_schedule = final_schedule.sort_values(
            by=["timeslot"], ascending=True)

        return final_schedule, df_x, df_y


def create_enrolment_df(He_s: np.array, S, E) -> pd.DataFrame:
    """
    Creates a dataframe with the students for each exam/course
    """
    exam_student_pairs = []
    for exam in range(len(He_s)):
        students_in_exam_e = []
        for i, student in enumerate(He_s[exam]):
            if student == 1:
                students_in_exam_e.append(S[i])
        exam_student_pairs.append(students_in_exam_e)

    enrolment_df = pd.DataFrame(columns=["EXAM", "student"])
    enrolment_df["EXAM"] = E
    enrolment_df["student"] = exam_student_pairs

    return enrolment_df
