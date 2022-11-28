
import pandas as pd
from datetime import datetime

unprocesses_filepath = "../database/"
processed_filepath   = "../database/processed/"

files_to_process = [
    "course_participants.csv",
    "courses.csv",
    "membership.csv",
    "session_participants.csv",
    "sessions.csv",
    "users.csv"
]

# Takes in a csv file and converts all specified columns to timestamps
def datetime_to_timestamp(csv_files):
    # Datetime columns to process from associated "files_to_process" array.
    columns_to_process = [
        ["course_bookingDate"],
        ["course_startTime", "course_endTime", "course_releaseTime"],
        ["membership_date"],
        ["session_bookingdate"],
        ["session_startTime", "session_endTime"],
        ["user_dateCreated"]
    ]

    for (filename, columns) in zip(files_to_process, columns_to_process):
        csv_file = csv_files[filename]
        for c in columns:
            for i in range(len(csv_file[c])):
                column_entry = csv_file[c][i]
                # Convert datetime to timestamp and replace entry with timestamp
                date = datetime.strptime(column_entry, '%Y-%m-%d %H:%M:%S')
                csv_file.loc[(i, c)] = str(int(datetime.timestamp(date)))

# NOTE: Dictionary of {course_type_string : integer_value}
def geneate_course_type_dictionary(course_types):
    course_dictionary = dict()
    for i in range(len(course_types)):
        course_dictionary[course_types[i]] = i

    # NOTE: I save the dict to a file because the integer values can change each time the data is processed.
    file = open("../database/processed/course_type_dictionary", "w")
    print(course_dictionary, file=file)
    file.close()
    
    return course_dictionary

def enumerate_course_types(course_csv_file):
    # Array of course types. The enumerated course type value will be the index of the list.
    course_types = list(set(course_csv_file["course_type"]))
    
    course_dictionary = geneate_course_type_dictionary(course_types)
    for i in range(len(course_csv_file["course_type"])):
        type = course_csv_file["course_type"][i]
        course_csv_file.loc[(i, "course_type")] = course_dictionary[type]

# Loads CSV files as a dictionary: {filename : csv_file}
def load_csv_files():
    csv_files = dict()
    for file in files_to_process:
        filepath = unprocesses_filepath + file
        csv_file = pd.read_csv(filepath, dtype=str, na_filter=False)
        csv_files[file] = csv_file
    return csv_files

def save_csv_files(csv_files):
    for (filename, location) in zip(files_to_process, files_to_process):
        file = csv_files[filename]
        file.fillna("NULL")
        file.to_csv(processed_filepath + location)

if __name__ == "__main__":
    csv_files = load_csv_files()

    # Process CSV files:
    datetime_to_timestamp(csv_files)
    enumerate_course_types(csv_files["courses.csv"])

    save_csv_files(csv_files)

