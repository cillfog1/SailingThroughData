
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
            year_label = c + "_year"
            month_label = c + "_month"
            date_label = c + "_date"
            for i in range(len(csv_file[c])):
                date = datetime.strptime(csv_file[c][i], '%Y-%m-%d %H:%M:%S')
                csv_file.loc[(i, year_label)]  = date.strftime('%Y')
                csv_file.loc[(i, month_label)] = date.strftime('%m')
                csv_file.loc[(i, date_label)]  = date.strftime('%d')

            csv_file.drop(columns=c, axis='columns', inplace=True)

# Replace string course type with an integer representation.
def enumerate_course_types(course_csv_file):
    course_type_dictionary = {'Adult Sailing': 0, 'First Aid': 1, 'Powerboating': 2, 'Private Lessons': 3, 'Sailing': 4, 'Try Sail': 5, 'Splash Club': 6}
    for i in range(len(course_csv_file["course_type"])):
        type = course_csv_file["course_type"][i]
        course_csv_file.loc[(i, "course_type")] = course_type_dictionary[type]

# Replace string course level with an integer representation.
def enumerate_course_levels(course_csv_file):
    sailing_level_dictionary = {'Taste of Sailing': 0, 'Start Sailing': 1, 'Basic Skills': 2, 'Basic / Improving': 3, 'Improving Skills': 4, 'Five Essentials Workshop': 5, 'Advanced Skills Workshop': 6, 'Improving / Advanced': 7, 'Advanced Boat Handling': 8, 'Blasket Trip': 9, 'Sacred Heart University': 10, 'Pobalscoil Chorca Dhuibhne': 11}
    powerboating_level_dictionary = {'Introduction to Powerboating': 0, 'National Powerboat Certificate': 1, 'Intermediate Powerboat Certificate': 2, 'Advanced Powerboat Certificate': 3}

    splashClub_level_dictionary = {'': 0, 'Private': 1}
    firstAid_level_dictionary = {'Basic': 0, 'Basic / Emergency': 1, 'Emergency': 2}

    for i in range(len(course_csv_file["course_level"])):
        type = course_csv_file["course_type"][i]
        # Remove trailing zeros
        level = course_csv_file["course_level"][i].strip()

        # If Adult Sailing or Sailing
        if (type == 0 or type == 4):
            course_csv_file.loc[(i, "course_level")] = sailing_level_dictionary[level]
        # If Powerboating
        elif (type == 2):
            course_csv_file.loc[(i, "course_level")] = powerboating_level_dictionary[level]
        # If First Aid
        elif (type == 1):
            course_csv_file.loc[(i, "course_level")] = firstAid_level_dictionary[level]
        # If Splash Club
        elif (type == 6):
            course_csv_file.loc[(i, "course_level")] = splashClub_level_dictionary[level]
        # Defaukt
        else:
            course_csv_file.loc[(i, "course_level")] = 0

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
        
# NOTE: After doing the csv file edits, a new index column is added at the from of the files.
# This removes it from all csv files.
def strip_files_csv_index():
    for filename in files_to_process:
        lines = []
        new_lines = []

        file = open("../database/processed/"+filename, "r", encoding='utf-8')
        while True:
            line = file.readline()
            if not line:
                break
            lines.append(line)
        file.close()

        for line in lines:
            new_start = line.find(',') + 1
            new_lines.append(line[new_start:])

        file = open("../database/processed/"+filename, "w")
        for line in new_lines:
            file.write(line)
        file.close()


# Loads CSV files as a dictionary: {filename : csv_file}
def load_csv_files():
    csv_files = dict()
    for file in files_to_process:
        filepath = "../database/" + file
        csv_file = pd.read_csv(filepath, dtype=str, na_filter=False)
        csv_files[file] = csv_file
    return csv_files

def save_csv_files(csv_files):
    for filename in files_to_process:
        file = csv_files[filename]
        file.fillna("NULL")
        file.to_csv("../database/processed/" + filename)


if __name__ == "__main__":
    csv_files = load_csv_files()

    # Process CSV files:
    datetime_to_timestamp(csv_files)
    enumerate_course_types(csv_files["courses.csv"])
    enumerate_course_levels(csv_files["courses.csv"])

    enumerate_user_sailing_cert(csv_files["users.csv"])
    enumerate_user_swimming_ability(csv_files["users.csv"])
    user_gender_to_binary(csv_files["users.csv"])
    user_birthdate_to_year(csv_files["users.csv"])

    save_csv_files(csv_files)
    strip_files_csv_index()

