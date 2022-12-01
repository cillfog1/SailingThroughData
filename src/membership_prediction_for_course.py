
# Predict the number of memberships for a particular course, in order to
# do this we are planning on doing multiple linear regressions
# [ Features ], [ Target ]

# Features:
# course_type.course_level, course_startDate

# Target:
# The number of memberships for that course

import pandas as pd

# ----------------------------------------------- Course Data -----------------------------------------------
# Course_Dat object to store the information on each course
class Course_Data:
    def __init__(self, course_id, course_type, course_level, target):
        self.course_id = course_id
        self.course_type = course_type
        self.course_level = course_level
        self.numOfMembers = target


# ----------------------------------------------- Extract Data -----------------------------------------------
# The file names to extract data from
csv_file_names = [
    "course_participants.csv",
    "courses.csv",
    "membership.csv"
]

#
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

        # Get number of users who are members for this course
        target = getNumberOfMembersForCourse(course_id)

        # Add courseData to courses
        currentCourseData = Course_Data(course_id, course_type, course_level, target)
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


# ----------------------------------------------- Main -----------------------------------------------
if __name__ == "__main__":
    csv_files = load_csv_files()
    courses = extractData(csv_files)

    for course in courses:
        print(str(course.course_id) + " : " + str(course.course_type) + "." + str(course.course_level) + " = " + str(course.numOfMembers))