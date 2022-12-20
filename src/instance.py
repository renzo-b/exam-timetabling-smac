import random
from typing import List

import numpy as np
import pandas as pd

# Global instance space
""" 
Training instances SMAC -> used to find the best CPLEX config
{'a_len_20189': 5262,
 'a_len_20191': 5189,
 'a_len_20195': 496,
 'a_len_20199': 5316,
 'a_len_20201': 5205,
 'a_len_20205': 1157,
 'a_len_20215': 771,
  'a_len_20221': 5510,
  a_len_20231': 5861
  
  
  
a_[20231, 20191]: 10063
a_[20231, 20199]: 9097
a_[20231, 20201]: 9005
a_[20221, 20199]: 7878
a_[20221, 20231]: 7044
a_[20221, 20201]: 7782
a_[20191, 20221]: 8859
"""

"""
Test instances: run best CPLEX -vs- default
 'a_len_20219': 5620,
 'a_len_20225': 1111,
 'a_len_20229': 5916,
 }
"""


merged_semester = {
    "a_[20231, 20191]": 10063,
    "a_[20231, 20199]": 9097,
    "a_[20231, 20201]": 9005,
    "a_[20221, 20199]": 7878,
    "a_[20221, 20231]": 7044,
    "a_[20221, 20201]": 7782,
    "a_[20191, 20221]": 8859,
}

student_per_semester = {
    "a_20189": [3000, 5262],
    "a_20191": [3000, 5189],
    "a_20195": [300, 496],
    "a_20199": [3000, 5316],
    "a_20201": [3000, 5205],
    "a_20205": [900, 1157],
    "a_20215": [600, 771],
    "a_20219": [3000, 5620],
    "a_20221": [3000, 5510],
    "a_20225": [900, 1111],
    "a_20229": [3000, 5916],
    "a_20231": [3000, 5861],
}

SEMESTERS = [
    "a_20189",
    "a_20191",
    # "a_20195",
    "a_20199",
    "a_20201",
    # "a_20205",
    # "a_20215",
]
TEST_SEMESTERS = [
    "a_20219",
    "a_20221",
    "a_20225",
    "a_20229",
    "a_20231",
]

NUMBER_INSTANCES = 1
# semester = "a_20195"
INSTANCE_SPACE = [
    {
        "num_students":
        # int(
        # np.linspace(
        # merged_semester[semester],
        # student_per_semester[semester][0],
        student_per_semester[semester][1],
        # NUMBER_INSTANCES,
        # )[i]
        # ),
        "every_n_room": 1,
        "np_seed": i,
        "room_avail_p": (np.linspace(99, 95, NUMBER_INSTANCES) / 100)[i],
        "prof_avail_p": (np.linspace(99, 95, NUMBER_INSTANCES) / 100)[i],
        "semester_date": semester,
    }
    for semester in SEMESTERS
    for i in range(NUMBER_INSTANCES)
]

TEST_INSTANCE_SPACE = [
    {
        "num_students": student_per_semester[semester_t][1],
        "every_n_room": 1,
        "np_seed": 1,
        "room_avail_p": 1,
        "prof_avail_p": 1,
        "semester_date": semester_t,
    }
    for semester_t in TEST_SEMESTERS
]


def get_dataset(
    num_students, every_n_room, np_seed, room_avail_p, prof_avail_p, semester_date
):
    file = "Exam Sched Prog Datasets.xlsx"
    rooms = pd.read_excel(file, sheet_name="datasets room caps ")
    schedule_20221 = pd.read_excel(file, sheet_name="20221 Final Schedule_fromLSM")

    hist_anonymized_enrolments = pd.read_excel(
        file, sheet_name="hist anonymized enrolments"
    )

    schedule_20221["Examdate"] = schedule_20221["Examdate"].astype(str)
    schedule_20221["Begin"] = schedule_20221["Begin"].astype(str)
    examdate_time_list = (
        (
            schedule_20221["Examdate"].dropna().astype(str)
            + " : "
            + schedule_20221["Begin"]
        )
        .dropna()
        .tolist()
    )

    ETL = np.unique(examdate_time_list)
    examdate_time_list = [i for i in ETL if i != "NaT : nan"]

    # new processing
    rooms["Room"] = rooms["Room"].astype(str)
    rooms["Room"] = rooms["Bd"].str.cat(rooms["Room"], sep=" ")

    ava_rooms = rooms["Room"].values
    ava_room_cap = rooms["Room Cap"].values
    hist_anonymized_enrolments["ACAD_ACT_CD"] = hist_anonymized_enrolments[
        "ACAD_ACT_CD"
    ].astype(str)
    hist_anonymized_enrolments["ACAD_ACT_CD"] = hist_anonymized_enrolments[
        "ACAD_ACT_CD"
    ].str.cat(hist_anonymized_enrolments["SECTION_CD"])

    semeser_list = np.unique(hist_anonymized_enrolments["SESSION_CD"]).tolist()
    anan_sem = {}
    for sem in semeser_list:
        anan_sem["a_{}".format(sem)] = hist_anonymized_enrolments[
            hist_anonymized_enrolments["SESSION_CD"] == sem
        ]

    # large_semester_list = [
    #     [20231, 20191],
    #     [20231, 20199],
    #     [20231, 20201],
    #     [20221, 20199],
    #     [20221, 20231],
    #     [20221, 20201],
    #     [20191, 20221],
    # ]

    # anan_sem = {}
    # name_list = []
    # for merge_sem in large_semester_list:
    #     name = "a_{}".format(merge_sem)
    #     name_list.append(name)
    #     anan_sem[name] = hist_anonymized_enrolments[
    #         (hist_anonymized_enrolments["SESSION_CD"] == merge_sem[0])
    #         | (hist_anonymized_enrolments["SESSION_CD"] == merge_sem[1])
    #     ]

    semester_df = anan_sem[semester_date]

    split = True

    while split == True:
        split = False
        for course in semester_df.groupby("ACAD_ACT_CD").size().index:
            course_size = semester_df.groupby("ACAD_ACT_CD").size().loc[course]
            if course_size > max(ava_room_cap):
                split = True
                idx_students_in_course = semester_df[
                    semester_df["ACAD_ACT_CD"] == course
                ].index
                half_students = int(len(idx_students_in_course) / 2)
                semester_df.loc[
                    idx_students_in_course[:half_students], "ACAD_ACT_CD"
                ] = (course + "_1")
                semester_df.loc[
                    idx_students_in_course[half_students:], "ACAD_ACT_CD"
                ] = (course + "_2")

    def processing_fun(semester, num_student):
        name = []
        exam_list = []
        Student_ID_Exams = []

        qwe = semester["HASHED_PERSON_ID"].values
        qwer = np.unique(qwe)
        random_sample_names_generator = (qwer[i] for i in range(num_student))

        for i in random_sample_names_generator:
            name.append(i)

        Student_ID_List = random.sample(name, len(name))

        for ID in Student_ID_List:
            Student_ID_Exams.append(
                semester["ACAD_ACT_CD"][semester.eq(ID).any(1)].values
            )

        for exam in Student_ID_Exams:
            exam_list.append(exam.any())

        uniq_exams = np.unique(exam_list)
        new_enrol = pd.DataFrame(semester).copy()
        new_enrol["class"] = 1

        temp_df = new_enrol[new_enrol["HASHED_PERSON_ID"].isin(Student_ID_List)]
        final_temp = temp_df[temp_df["ACAD_ACT_CD"].isin(uniq_exams)]

        courses_enrollments_set = pd.pivot_table(
            data=final_temp,
            values="class",
            index="ACAD_ACT_CD",
            columns=["HASHED_PERSON_ID"],
            fill_value=0,
        )

        return uniq_exams, Student_ID_List, courses_enrollments_set

    t_E, t_S, t_C = processing_fun(semester_df, num_students)

    # storing
    exam_set = t_E
    student_set = t_S
    datetime_slot_set = examdate_time_list  # list(range(64))  # examdate_time_list
    room_set = ava_rooms
    room_capacity_set = ava_room_cap
    courses_enrollments_set = t_C.values

    # extract every N room to reduce the number of variables and constraints
    room_set = room_set[::every_n_room]
    room_capacity_set = room_capacity_set[::every_n_room]

    # make some rooms unavailable with some probability
    np.random.seed(np_seed)
    room_availability = np.random.choice(
        2,
        size=(len(room_set), len(datetime_slot_set)),
        p=[room_avail_p, 1 - room_avail_p],
    )
    prof_availability = np.random.choice(
        2,
        size=(len(exam_set), len(datetime_slot_set)),
        p=[prof_avail_p, 1 - prof_avail_p],
    )

    return (
        exam_set,
        student_set,
        datetime_slot_set,
        room_set,
        room_capacity_set,
        courses_enrollments_set,
        room_availability,
        prof_availability,
        semester_date,
    )


def get_ET_instance(instance_num: int):

    instance_config = INSTANCE_SPACE[instance_num]

    # load dataset
    (
        exam_set,
        student_set,
        datetime_slot_set,
        room_set,
        room_capacity_set,
        courses_enrollments_set,
        room_availability,
        prof_availability,
        semester_date,
    ) = get_dataset(**instance_config)

    # Create instance
    instance = ET_Instance(
        exam_set,
        student_set,
        datetime_slot_set,
        room_set,
        room_capacity_set,
        courses_enrollments_set,
        1 / 60,
        room_availability,
        prof_availability,
        semester_date,
    )

    return instance


class ET_Instance:
    def __init__(
        self,
        exam_set: List,
        student_set: List,
        datetime_slot_set: List,
        room_set: List,
        room_capacity_set: List,
        courses_enrollments_set,
        ratio_inv_students: float,
        room_availability: float,
        prof_availability: float,
        semester_date: str,
    ):
        self.exam_set = exam_set
        self.student_set = student_set
        self.datetime_slot_set = datetime_slot_set
        self.room_set = room_set
        self.room_capacity_set = room_capacity_set
        self.courses_enrollments_set = courses_enrollments_set
        self.ratio_inv_students = ratio_inv_students
        self.room_availability = room_availability
        self.prof_availability = prof_availability
        self.semester_date = semester_date

        self.instance_code = f"{len(exam_set)}_{len(student_set)}_{len(datetime_slot_set)}_{len(room_set)}_{courses_enrollments_set.shape}_{semester_date}"

        print("generated a new IT instance")
        self.print_instance_info()

    def print_instance_info(self):
        print(f"Data from semester       : {self.semester_date}")
        print(f"Number of exams          : {len(self.exam_set)}")
        print(f"Number of students       : {len(self.student_set)}")
        print(f"Number of rooms          : {len(self.room_set)}")
        print(f"Number of datetime slots : {len(self.datetime_slot_set)}")
        print(f"Shape of enrollments     : {self.courses_enrollments_set.shape}")
        print(f"Ratio of inv/students    : {self.ratio_inv_students}")
