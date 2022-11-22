from typing import List

import numpy as np


def get_ET_instance(instance_num : int):
    instance = ET_Instance(
        exam_set = ['CSC101', 'CSC102', 'CSC103', 'CSC104', 'CSC111', 'CSC110'], 
        student_set = ['Aaron','Bruno','Cell','Dodo','Earl','Frank', 'Gary', 'Hilton', 'Ian'], 
        datetime_slot_set = ['Dec 1st 9am', 'Dec 1st 12pm', 'Dec 2nd 9am', 'Dec 2nd 12pm', 'Dec 3rd 9am'], 
        room_set = ['SB1', 'SB2','SB3','SB4', 'SB6','SB7'], 
        courses_enrollments_set = np.asarray([
            [0, 0, 1, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1],
        ]), 
        room_capacity_set = [1, 1, 1, 1, 1, 1], 
        ratio_inv_students = 1/3,
    )
    
    return instance


class ET_Instance:
    def __init__(self,
        exam_set : List, 
        student_set : List, 
        datetime_slot_set : List, 
        room_set : List, 
        room_capacity_set : List, 
        courses_enrollments_set, 
        ratio_inv_students : float,

    ):
        self.exam_set = exam_set
        self.student_set = student_set
        self.datetime_slot_set = datetime_slot_set
        self.room_set = room_set
        self.room_capacity_set = room_capacity_set
        self.courses_enrollments_set = courses_enrollments_set
        self.ratio_inv_students = ratio_inv_students

        print('generated a new IT instance')
        self.print_instance_info()

    def print_instance_info(self):  
        print(f"Number of exams          : {len(self.exam_set)}")
        print(f"Number of students       : {len(self.student_set)}")
        print(f"Number of rooms          : {len(self.room_set)}")
        print(f"Number of datetime slots : {len(self.datetime_slot_set)}")
        print(f"Shape of enrollments     : {self.courses_enrollments_set.shape}")    