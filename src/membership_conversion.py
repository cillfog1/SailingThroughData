
# Find any relationships in the conversions from students becoming members, in order to
# do this we are planning on doing multiple logistic regressions
# [ Features ], [ Label ]

# Features:
# OLD: How many courses they did in a year, if they are local
# NEW: How many courses they did in a year, swimming ability

# Labels:
# If they are a Student (-1) or a Member(1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing   import PolynomialFeatures
#from sklearn.linear_model    import LogisticRegression
#from sklearn.neighbors       import KNeighborsClassifier
#from sklearn.dummy           import DummyClassifier
#from sklearn.model_selection import KFold, cross_val_score
#from sklearn.metrics         import f1_score, confusion_matrix, roc_curve

csv_file_names = [
    "course_participants.csv",
    "courses.csv",
    "membership.csv",
    "session_participants.csv",
    "sessions.csv",
    "users.csv"
]

class Data:
    def __init__(self, user_ids, course_counts, swimming_abilities, sailing_certs, labels):
        self.user_ids           = user_ids
        self.course_counts      = course_counts
        self.swimming_abilities = swimming_abilities
        self.sailing_certs      = sailing_certs
        self.labels             = labels

#class Data:
#    def __init__(self, course_count, local, label):
#        self.course_count = course_count
#        self.local = local
#        self.label = label

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
def generate_student_or_member(year, user_ids, csv_files):
    # Set all user ids to False
    user_id_membership = dict()
    insert_user_ids(user_id_membership, user_ids)

    # Set all user ids who are members to True
    members = get_members(year, csv_files["membership.csv"])
    determine_user_members(user_id_membership, members)

    return user_id_membership

#def generate_user_local(user_ids, csv_file):
#    is_local_map = dict()
#    for user in user_ids:
#        is_local_map[user] = False
#
#    csv_is_local = csv_file["user_local"]
#    for user_id in user_ids:
#        is_local_map[int(user_id)] = csv_is_local[int(user_id)-1]
#
#    return is_local_map

def generate_course_count(year, user_ids, csv_file):
    # Get all indices where the year matches
    indices = []
    years = csv_file["course_bookingDate_year"]
    for (y, i) in zip(years, range(len(years))):
        if y == year:
            indices.append(i)

    # Generate how many times each user booked a course
    csv_user_ids = csv_file["user_id"]
    course_counts = []
    for user in user_ids:
        current_user_count = 1
        for i in indices:
            if csv_user_ids[i] == user:
                current_user_count += 1
        course_counts.append(current_user_count)

    user_id_course_count_map = {user_ids[i]: float(course_counts[i]) for i in range(len(user_ids))}
    return user_id_course_count_map

def generate_swimming_abilities(user_ids, csv_file):
    csv_abilities = csv_file["user_swimmingAbility"]
    swimming_abilities = dict()
    for user in user_ids:
        swimming_abilities[user] = float(csv_abilities[int(user)-1])
    return swimming_abilities

def generate_sailing_certs(user_ids, csv_file):
    csv_sailing_certs = csv_file["user_sailingCert"]
    sailing_certs = dict()
    for user in user_ids:
        sailing_certs[user] = float(csv_sailing_certs[int(user)-1])
    return sailing_certs


def generate_data(year, csv_files):
    user_ids = get_user_ids(year, csv_files["users.csv"])

    #user_is_local_map = generate_user_local(user_ids, csv_files["users.csv"])
    user_course_count     = generate_course_count(year, user_ids, csv_files["course_participants.csv"])
    user_swimming_ability = generate_swimming_abilities(user_ids, csv_files["users.csv"])
    user_sailing_cert     = generate_sailing_certs(user_ids, csv_files["users.csv"])
    user_is_member        = generate_student_or_member(year, user_ids, csv_files)
    
    data = Data(user_ids, user_course_count.values(), user_swimming_ability.values(), user_sailing_cert.values(), user_is_member.values())
    
    return data

def data_to_numpy_dataset(data):
    course_counts      = list(data.course_counts)
    swimming_abilities = list(data.swimming_abilities)
    sailing_certs      = list(data.sailing_certs)
    is_member_labels   = list(data.labels)
    print(type(sailing_certs[0]))
    dataframe = pd.DataFrame(list(zip(course_counts, swimming_abilities, sailing_certs, is_member_labels)), columns =['Course Count', 'Swimming Ability', 'Sailing Cert', 'Member'])
    return dataframe

def load_csv_files():
    csv_files = dict()
    for file in csv_file_names:
        filepath = "../database/processed/" + file
        csv_file = pd.read_csv(filepath, dtype=str, na_filter=False)
        csv_files[file] = csv_file
    return csv_files

def graph_data(dataset):
    course_counts = dataset.iloc[:,0]
    swimming_ability = dataset.iloc[:,1]
    y  = dataset.iloc[:,2]
    _, ax = plt.subplots()
    for yi in np.unique(y):
        ix = np.where(y == yi)[0]
        col = 'blue' if yi else 'lime'
        l = '-1 actual' if yi else '1 actual'
        ax.scatter(course_counts.take(ix), swimming_ability.take(ix), c=col, marker='o', label=l)
    plt.xlabel('Course Count')
    plt.ylabel('Swimming Ability')
    plt.legend(loc='upper right')
    plt.title('Scatter plot of Couse Count against Swimming Ability')
    #fig_name = 'dataset1.png' if first else 'dataset2.png'
    #plt.savefig(fig_name)
    plt.show()
    plt.clf()

def graph_data(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    course_counts = dataset.iloc[:,0]
    swimming_abilities = dataset.iloc[:,1]
    sailing_certs = dataset.iloc[:,2]
    y = dataset.iloc[:,3]
    for yi in np.unique(y):
        ix = np.where(y == yi)[0]
        col = 'blue' if yi else 'lime'
        l = '-1 actual' if yi else '1 actual'
        ax.scatter(course_counts.take(ix), swimming_abilities.take(ix), sailing_certs.take(ix), c=col, label=l)

    ax.set_xlabel('Course Count')
    ax.set_ylabel('Swimming Ability')
    ax.set_zlabel('Sailing Cert')
    plt.title('3D Scatter Plot of Data')
    plt.show()

if __name__ == "__main__":
    csv_files = load_csv_files()
    data = generate_data("2020", csv_files)
    dataset = data_to_numpy_dataset(data)
    graph_data(dataset)

