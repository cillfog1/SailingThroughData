
import pandas as pd
from datetime import datetime

# Takes in a csv file and converts all specified columns to timestamps
def datetime_to_timestamp(csv_file, column_names):
    for c in column_names:
        for i in range(len(csv_file[c])):
            column_entry = csv_file[c][i]
            # Convert datetime to timestamp and replace entry with timestamp
            date = datetime.strptime(column_entry, '%Y-%m-%d %H:%M:%S')
            csv_file.loc[(i, c)] = str(int(datetime.timestamp(date)))

if __name__ == "__main__":
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
        filepath = unprocesses_filepath + filename
        csv_file = pd.read_csv(filepath, dtype=str, na_filter=False)

        # NOTE: More data pre-processing can be done on the csv file after this.
        datetime_to_timestamp(csv_file, columns)

        csv_file.fillna("NULL")
        # Save CSV file.
        finished_filepath = processed_filepath + filename
        csv_file.to_csv(finished_filepath)

