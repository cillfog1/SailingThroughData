
import pandas as pd
from datetime import datetime

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

# Replace string course type with an integer representation.
def enumerate_course_types(course_csv_file):
    course_dictionary = {'Adult Sailing': 0, 'First Aid': 1, 'Powerboating': 2, 'Private Lessons': 3, 'Sailing': 4, 'Try Sail': 5, 'Splash Club': 6}
    for i in range(len(course_csv_file["course_type"])):
        type = course_csv_file["course_type"][i]
        course_csv_file.loc[(i, "course_type")] = course_dictionary[type]

# Replace sailing cert with integer representation.
def enumerate_user_sailing_cert(csv_file):
    # NOTE: If there is no cert provided I am assuming there is none. This may be a bad/incorrect assumption.
    sailing_cert_dict = {'': 0, 'None': 0, 'Start Sailing': 1, 'Taste of Sailing': 2, 'Basic Skills': 3, 'Improving Skills': 4, 'Improving Skills+': 5}
    for i in range(len(csv_file["user_sailingCert"])):
        cert_str = csv_file["user_sailingCert"][i]
        csv_file.loc[(i, "user_sailingCert")] = sailing_cert_dict[cert_str]

# Replace swimming ability with integer representation.
def enumerate_user_swimming_ability(csv_file):
    ability_dict = {'None': 0, 'Poor': 1, 'Fair': 2, 'Good': 3}
    for i in range(len(csv_file["user_swimmingAbility"])):
        ability = csv_file["user_swimmingAbility"][i]
        csv_file.loc[(i, "user_swimmingAbility")] = ability_dict[ability]

# NOTE: All 'Other's are not Male.
def user_gender_to_binary(csv_file):
    gender_dic = {'Female': 0, 'Male': 1, 'Other': 1}
    for i in range(len(csv_file["user_gender"])):
        gender_str = csv_file["user_gender"][i]
        csv_file.loc[(i, "user_gender")] = gender_dic[gender_str]

# NOTE: This was to timestamp but there were people born before 1970 and therefore gave negative values.
def user_birthdate_to_year(csv_file):
    dates = csv_file["user_dob"]
    for i in range(len(dates)):
        date = datetime.strptime(dates[i], '%Y-%m-%d')
        csv_file.loc[(i, "user_dob")] = date.strftime("%Y")
        

# Loads CSV files as a dictionary: {filename : csv_file}
def load_csv_files():
    csv_files = dict()
    for file in files_to_process:
        filepath = "../database/" + file
        csv_file = pd.read_csv(filepath, dtype=str, na_filter=False)
        csv_files[file] = csv_file
    return csv_files

def save_csv_files(csv_files):
    for (filename, location) in zip(files_to_process, files_to_process):
        file = csv_files[filename]
        file.fillna("NULL")
        file.to_csv("../database/processed/" + location)


if __name__ == "__main__":
    csv_files = load_csv_files()

    # Process CSV files:
    datetime_to_timestamp(csv_files)
    enumerate_course_types(csv_files["courses.csv"])

    enumerate_user_sailing_cert(csv_files["users.csv"])
    enumerate_user_swimming_ability(csv_files["users.csv"])
    user_gender_to_binary(csv_files["users.csv"])
    user_birthdate_to_year(csv_files["users.csv"])

    save_csv_files(csv_files)

