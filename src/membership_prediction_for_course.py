
# Predict the number of memberships for a particular course, in order to
# do this we are planning on doing multiple linear regressions
# [ Features ], [ Target ]

# Features:
# course_type.course_level, course_startDate

# Target:
# The number of memberships for that course

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------- Course Data -----------------------------------------------
# Course_Dat object to store the information on each course
class Course_Data:
    def __init__(self, course_id, course_type, course_level, course_startMonth, course_startDate, target):
        self.course_id = course_id
        self.course_type = course_type
        self.course_level = course_level
        self.course_startMonth = course_startMonth
        self.course_startDate = course_startDate
        self.numOfMembers = target


# ----------------------------------------------- Extract Data -----------------------------------------------
# The file names to extract data from
csv_file_names = [
    "course_participants.csv",
    "courses.csv",
    "membership.csv"
]

def load_csv_files():
    csv_files = dict()
    for file in csv_file_names:
        filepath = "../database/processed/" + file
        csv_file = pd.read_csv(filepath, dtype=str, na_filter=False)
        csv_files[file] = csv_file
    return csv_files

def extractData(csv_files):
    courses = []

    # Extract data from courses.csv
    courseFile = csv_files["courses.csv"]
    for i in range(len(courseFile)):
        course_id = int(courseFile["course_id"][i])
        course_type = int(courseFile["course_type"][i])
        course_level = int(courseFile["course_level"][i])
        course_startMonth = int(courseFile["course_startTime_month"][i])
        course_startDate = int(courseFile["course_startTime_date"][i])

        # Get number of users who are members for this course
        target = getNumberOfMembersForCourse(course_id)

        # Add courseData to courses
        currentCourseData = Course_Data(course_id, course_type, course_level, course_startMonth, course_startDate, target)
        courses.append(currentCourseData)
    return courses

def getNumberOfMembersForCourse(course_id):
    courseParticpantsFile = csv_files["course_participants.csv"]
    numOfMembers = 0
    for i in range(len(courseParticpantsFile)):
        currentCourse_id = int(courseParticpantsFile["course_id"][i])
        if currentCourse_id == course_id:
            numOfMembers += 1
    return numOfMembers

def convertToSeperateArrays(courses):
    course_level_type = []
    course_startDate = []
    course_numOfMembers = []
    for course in courses:
        course_type_level_as_float = float( str(course.course_type) + "." + str(course.course_level) )
        course_level_type.append(course_type_level_as_float)

        date_as_percent_of_month = (1/31) * course.course_startDate
        course_startDate_as_float = course.course_startMonth + date_as_percent_of_month
        course_startDate.append(course_startDate_as_float)

        course_numOfMembers.append(course.numOfMembers)
    return course_level_type, course_startDate, course_numOfMembers


# ----------------------------------------------- 3D Scatter Plot Data -----------------------------------------------
def threeDScatterPlot(feature1, feature2, target):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(feature1, feature2, target)
    ax.set_xlabel('course_type.course_level')
    ax.set_ylabel('course_startDate')
    ax.set_zlabel('number_of_members')

    plt.title('3D Scatter Plot of Data')
    plt.show()


# ----------------------------------------------- Main -----------------------------------------------
if __name__ == "__main__":

    # Extract Data
    csv_files = load_csv_files()
    courses = extractData(csv_files)

    # Convert courses to seperate arrays
    course_level_type, course_startDate, course_numOfMembers = convertToSeperateArrays(courses)

    # 3D Scatter Plot of the data
    threeDScatterPlot(course_level_type, course_startDate, course_numOfMembers)