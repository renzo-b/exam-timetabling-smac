import random
from typing import List

import numpy as np
import pandas as pd


def get_dataset(num_students):
    file = "Exam Sched Prog Datasets.xlsx"
    rooms = pd.read_excel(file, sheet_name = "datasets room caps ")
    courses = pd.read_excel(file, sheet_name = "20221 course size")
    enrolments = pd.read_excel(file, sheet_name = "20221 anonymized enrolments")
    schedule = pd.read_excel(file, sheet_name = "20221 Final Schedule_fromLSM")
    OG_Course_list = schedule.Course.dropna()
    OG_Course_list = OG_Course_list.values
    
    # Data Extraction
    #### Random sample of ID's
    qwe = enrolments["HASHED_PERSON_ID"].values
    qwer = np.unique(qwe)
    #random_sample_names_generator = (qwer[i] for i in range(np.random.randint(0, len(qwer)-1)))
    random_sample_names_generator = (qwer[i] for i in range(num_students))
    name = []
    
    for i in random_sample_names_generator:
        name.append(i)

    Student_ID_List = random.sample(name, len(name))
    print(len(Student_ID_List))
    
    #### Unique Exams for given student
    Student_ID_Exams = []
    #enrolments["HASHED_PERSON_ID"].loc[ID]
    for ID in Student_ID_List:
        Student_ID_Exams.append(enrolments["ACAD_ACT_CD"][enrolments.eq(ID).any(1)].values)
    
    exam_list = []
    for exam in Student_ID_Exams:
        exam_list.append(exam.any())
    uniq_exams = np.unique(exam_list)
    
    new_enrol = pd.DataFrame(enrolments).copy()
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
    idlist = enrolments['HASHED_PERSON_ID'].tolist()
    classlist = enrolments['ACAD_ACT_CD'].tolist()

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
    course_enrollment_values = courses_enrollments_set.drop(["TEP445H1"], axis=0).values
    sumHe_s = np.sum(course_enrollment_values, axis=1)

    room_list = schedule['bd_room'].dropna().values
    test_list = room_list.tolist()
    roomlist = [i for i in test_list if i != "None"]
    
    room_cap_list = schedule['Room Cap'].dropna().values
    test_cap_list = room_cap_list.tolist()
    roomcaplist = [i for i in test_cap_list if i != "None"]
    
    exam_set = schdcrselist
    student_set = Student_ID_List
    datetime_slot_set = examdate_time_list
    room_set = roomlist
    room_capacity_set = roomcaplist
    courses_enrollments_set = course_enrollment_values

    return (exam_set, student_set, datetime_slot_set, room_set, room_capacity_set, 
        courses_enrollments_set)


def get_ET_instance(instance_num : int):
<<<<<<< Updated upstream

    student_number_combinations = [500, 1000, 3000]
    num_students = student_number_combinations[instance_num]
    
    (exam_set, student_set, datetime_slot_set, room_set, 
    room_capacity_set, courses_enrollments_set) = get_dataset(num_students)
    
    instance = ET_Instance(exam_set, student_set, datetime_slot_set, room_set, 
    room_capacity_set, courses_enrollments_set, 1/60)
=======
    instance = ET_Instance(
        exam_set =  ['CSC101', 'CSC102', 'CSC103', 'CSC104', 'CSC111', 'CSC110'], 
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
>>>>>>> Stashed changes
    
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