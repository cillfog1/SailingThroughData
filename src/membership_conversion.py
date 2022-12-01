
# Find any relationships in the conversions from students becoming members, in order to
# do this we are planning on doing multiple logistic regressions
# [ Features ], [ Label ]

# Features:
# How many courses they did in a year, if they are local

# Labels:
# If they are a Student (-1) or a Member(1)

import pandas as pd

csv_file_names = [
    "course_participants.csv",
    "courses.csv",
    "membership.csv",
    "session_participants.csv",
    "sessions.csv",
    "users.csv"
]

class User_Data:
    def __init__(self, user_id, course_count, local, label):
        self.user_id = user_id
        self.course_count = course_count
        self.local = local
        self.label = label

def get_user_ids(year, user_csv):
    # Get all indices where the year matches
    indices = []
    years = user_csv["user_dateCreated_year"]
    for (y, i) in zip(years, range(len(years))):
        if y == year:
            indices.append(i)

    # Get all user ids using indices
    csv_user_ids = user_csv["user_id"]
    user_ids = []
    for i in indices:
        user_ids.append(csv_user_ids[i])

    return user_ids

def get_members(year, membership_csv):
    # Get all indices where the year matches
    indices = []
    years = membership_csv["membership_year"]
    for (y, i) in zip(years, range(len(years))):
        if y == year:
            indices.append(i)

    # Get all member for that year using indices
    members_list = membership_csv["user_id"]
    members = []
    for i in indices:
        members.append(members_list[i])

    return members

def insert_user_ids(user_membership_map, user_ids):
    for user_id in user_ids:
        user_membership_map[user_id] = False

def determine_user_members(user_membership_map, member_list):
    for m in member_list:
        user_membership_map[m] = True

# NOTE: Returns a map of {user_id : is_member}
# NOTE: This does not work for years other than 2020. This is because if we want to look at members for year
# 2021, I would look at the user accounts created in 2021 and check the membership table to see if they are a
# member. What I should do as well I check all user account that were created in the years previous to this
# as well.
def generate_student_or_member(year, csv_files):
    user_id_membership = dict()

    user_ids = get_user_ids(year, csv_files["users.csv"])
    insert_user_ids(user_id_membership, user_ids)

    members = get_members(year, csv_files["membership.csv"])
    determine_user_members(user_id_membership, members)

    return user_id_membership

def load_csv_files():
    csv_files = dict()
    for file in csv_file_names:
        filepath = "../database/processed/" + file
        csv_file = pd.read_csv(filepath, dtype=str, na_filter=False)
        csv_files[file] = csv_file
    return csv_files

if __name__ == "__main__":
    csv_files = load_csv_files()
    is_member = generate_student_or_member("2020", csv_files)
    print(is_member)

