import random
from typing import List

import numpy as np
import pandas as pd

### Global instance space
INSTANCE_SPACE = [
    #{"num_students":500, "every_n_room":10, "np_seed":0, "room_avail_p":0.95, "prof_avail_p":0.95},
    #{"num_students":800, "every_n_room":10, "np_seed":1, "room_avail_p":0.95, "prof_avail_p":0.95},
    {"num_students":5000, "every_n_room":10, "np_seed":1, "room_avail_p":0.95, "prof_avail_p":0.95},
]

def get_dataset(num_students, every_n_room, np_seed, room_avail_p, prof_avail_p):
    file = "Exam Sched Prog Datasets.xlsx"
    #file = r"C:\Users\William Hazen\Documents\GitHub\exam-timetabling-smac\Exam Sched Prog Datasets.xlsx"
    rooms = pd.read_excel(file, sheet_name = "datasets room caps ")
    courses = pd.read_excel(file, sheet_name = "20221 course size")
    enrolments = pd.read_excel(file, sheet_name = "20221 anonymized enrolments")
    schedule = pd.read_excel(file, sheet_name = "20221 Final Schedule_fromLSM")
    his_enrolments = pd.read_excel(file, sheet_name = "hist anonymized enrolments")
    OG_Course_list = schedule.Course.dropna()
    OG_Course_list = OG_Course_list.values
    
    # Data Extraction
    #### Random sample of ID's
    qwe = his_enrolments["HASHED_PERSON_ID"].values
    qwer = np.unique(qwe)
    #random_sample_names_generator = (qwer[i] for i in range(np.random.randint(0, len(qwer)-1)))
    random_sample_names_generator = (qwer[i] for i in range(num_students))
    name = []
    
    for i in random_sample_names_generator:
        name.append(i)

    Student_ID_List = random.sample(name, len(name))
    
    #### Unique Exams for given student
    Student_ID_Exams = []
    for ID in Student_ID_List:
        Student_ID_Exams.append(his_enrolments["ACAD_ACT_CD"][his_enrolments.eq(ID).any(1)].values)
    
    exam_list = []
    for exam in Student_ID_Exams:
        exam_list.append(exam.any())
    uniq_exams = np.unique(exam_list)
    
    new_enrol = pd.DataFrame(his_enrolments).copy()
    new_enrol["class"] = 1
    temp_df = new_enrol[new_enrol["HASHED_PERSON_ID"].isin(Student_ID_List)]
    final_temp = temp_df[temp_df["ACAD_ACT_CD"].isin(uniq_exams)]
    
    #### Course Enrollment
    
    courses_enrollments_set = pd.pivot_table(data=final_temp, values="class", index="ACAD_ACT_CD", columns=["HASHED_PERSON_ID"], fill_value=0)
    # Cleaning Room Table
    #Cleaning room table 

    rooms['Room'] = rooms['Room'].astype(str)
    rooms['Room'] = rooms['Bd'].str.cat(rooms['Room'], sep = " ")

    rooms = rooms.drop(['Note','Bd'], axis=1)
    # Cleaning Course Data
    #Cleaning course data

    courses['COURSE_CODE'] = courses['COURSE_CODE'].str.cat(courses['SECTIONCD'], sep = "")
    courses = courses.drop(['SESSION_CD','SECTIONCD','ADMINFACULTYCODE','ADMINDEPT'], axis=1)
    crselist = courses['COURSE_CODE'].tolist()
    sizelist = courses['CURR_REG_QTY'].tolist()
    # Enrolment data
    idlist = his_enrolments['HASHED_PERSON_ID'].tolist()
    classlist = his_enrolments['ACAD_ACT_CD'].tolist()

    schedule['Room'] = schedule['Room'].astype(str)
    schedule['Examdate'] = schedule['Examdate'].astype(str)
    schedule['Begin'] = schedule['Begin'].astype(str)
    schedule['Ends'] = schedule['Ends'].astype(str)

    schedule['Room'] = schedule['Bd'].str.cat(schedule['Room'], sep = " ")

    schedule = schedule.drop(['Bd','Faculty','Department','Section','Sesson','Course Name','Duration'], axis=1)

    schedule.replace('ZZ REFER_TO_FACULTY_SCHEDULE', 'None', inplace=True)
    schedule.replace('ZZ REFER_TO_FACULTY_SCHEDULE', 'None', inplace=True)
    schedule.replace('ZZ ONLIN', 'None', inplace=True)
    schedule.replace('APSCDept APSCDept', 'None', inplace=True)
    schedule.replace('APSCDept ComputerLab', 'None', inplace=True)
    schedule.replace('ZZ KNOX', 'None', inplace=True)
    schedule.replace('ZZ VLAD', 'None', inplace=True)
    schedule.replace('ZZ ONLINMUSIC', 'None', inplace=True)

    OG_CL = list(OG_Course_list)

    unique_course_list=[]
    for course in OG_CL:
        for crs in uniq_exams:
            if course[0:8] == crs:
                unique_course_list.append(course) 

    new_schedule = pd.DataFrame(schedule).copy()
    temp_sch_df = new_schedule[new_schedule["Course"].isin(unique_course_list)]
    #Schedule data in list format

    schdcrselist = temp_sch_df['Course'].tolist()
    schdroomlist = temp_sch_df['Room'].tolist()
    exmdatelist = temp_sch_df['Examdate'].tolist()
    starttimelist = temp_sch_df['Begin'].tolist()
    endtimelist = temp_sch_df['Ends'].tolist()
    enrllist = temp_sch_df['Enrolment'].tolist()
    #crseinroomlist = temp_sch_df['Course In Room'].tolist()
    #roomcaplist = temp_sch_df['Room Cap'].tolist()
    totalocclist = temp_sch_df['Total Occupancy'].tolist()
    examdate_time_list = (schedule['Examdate'].dropna().astype(str) + " : " + schedule['Begin']).dropna().tolist()
    ETL = np.unique(examdate_time_list)
    examdate_time_list = [i for i in ETL if i != "NaT : nan"]
    course_enrollment_values = courses_enrollments_set.values #.drop(["TEP445H1"], axis=0).values
    sumHe_s = np.sum(courses_enrollments_set, axis=1)

    room_list = schedule['bd_room'].dropna().values
    test_list = room_list.tolist()
    roomlist = [i for i in test_list if i != "None"]
    
    room_cap_list = schedule['Room Cap'].dropna().values
    test_cap_list = room_cap_list.tolist()
    roomcaplist = [i for i in test_cap_list if i != "None"]
    
    exam_set = uniq_exams
    student_set = Student_ID_List
    datetime_slot_set = examdate_time_list
    room_set = roomlist
    room_capacity_set = roomcaplist
    courses_enrollments_set = course_enrollment_values

    # extract every N room to reduce the number of variables and constraints
    room_set = room_set[::every_n_room]
    room_capacity_set = room_capacity_set[::every_n_room]

    np.random.seed(np_seed)
    room_availability = np.random.choice(
        2, size=(len(room_set), len(datetime_slot_set)), 
        p=[room_avail_p, 1- room_avail_p])
    prof_availability = np.random.choice(
        2, size=(len(exam_set), len(datetime_slot_set)), 
        p=[prof_avail_p, 1- prof_avail_p])



    return (exam_set, student_set, datetime_slot_set, room_set, room_capacity_set, 
        courses_enrollments_set, room_availability, prof_availability)

def get_ET_instance(instance_num : int):   

    instance_config = INSTANCE_SPACE[instance_num]

    # load dataset
    (exam_set, student_set, datetime_slot_set, room_set, 
    room_capacity_set, courses_enrollments_set, 
    room_availability, prof_availability) = get_dataset(**instance_config)
    
    # Create instance
    instance = ET_Instance(exam_set, student_set, datetime_slot_set, room_set, 
    room_capacity_set, courses_enrollments_set, 1/60, room_availability, prof_availability)
    
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
        room_availability : float,
        prof_availability : float,
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

        print('generated a new IT instance')
        self.print_instance_info()

    def print_instance_info(self):  
        print(f"Number of exams          : {len(self.exam_set)}")
        print(f"Number of students       : {len(self.student_set)}")
        print(f"Number of rooms          : {len(self.room_set)}")
        print(f"Number of datetime slots : {len(self.datetime_slot_set)}")
        print(f"Shape of enrollments     : {self.courses_enrollments_set.shape}")    