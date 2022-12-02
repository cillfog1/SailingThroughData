
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.linear_model    import LogisticRegression
from sklearn.dummy           import DummyClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics         import f1_score, confusion_matrix, roc_curve

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

def noramlise(X):
    shift = np.average(X)
    scalingFactor = np.max(X) - np.min(X)
    X = (X-shift) / scalingFactor
    return X

def data_to_numpy_dataset(course_count_data, ability_data, is_member):
    dataframe = pd.DataFrame(list(zip(course_count_data, ability_data, is_member)), columns =['Course Count', 'Swimming Ability', 'Member'])
    return dataframe

def load_csv_files():
    csv_files = dict()
    for file in csv_file_names:
        filepath = "../database/processed/" + file
        csv_file = pd.read_csv(filepath, dtype=str, na_filter=False)
        csv_files[file] = csv_file
    return csv_files

# Course Count Range    : 1 - 4
# Swimming Ability Range: 0 - 3
# Sailing Ability Range : 0 - 5
# I want to find how many users have a go to a given number of courses and have a particular swimming ability
# NOTE: Course count is shifted down by -1 to match the index ranges of swimming and sailing ability.
def generate_scatter_marker_size(data):
    user_ids = list(data.user_ids)
    counts = list(data.course_counts)
    swimming = list(data.swimming_abilities)
    sailing = list(data.sailing_certs)

    count_values = set(counts)
    swimming_values = set(swimming)
    sailing_values = set(sailing)

    count_size = len(count_values)
    swimming_size = len(swimming_values)
    sailing_size = len(sailing_values)

    counts_swimming_sizes = [[0 for _ in range(swimming_size)] for _ in range(count_size)]
    counts_sailing_sizes = [[0 for _ in range(sailing_size)] for _ in range(count_size)]

    for c in count_values:
        for s in swimming_values:
            count = 0
            for user in range(len(user_ids)):
                index = user - 1
                if counts[index]-1 == c and swimming[index] == s:
                    count += 1
            counts_swimming_sizes[c-1][s] = count

    for c in count_values:
        for s in sailing_values:
            count = 0
            for user in range(len(user_ids)):
                index = user - 1
                if counts[index]-1 == c and sailing[index] == s:
                    count += 1
            counts_sailing_sizes[c-1][s] = count

    return counts_swimming_sizes, counts_sailing_sizes

def graph_2D_data(x1, x2, y, x1_label, x2_label):
    _, ax = plt.subplots()
    for yi in np.unique(y):
        ix = np.where(y == yi)[0]
        col = 'blue' if yi else 'lime'
        l = 'Member' if yi else 'Student'
        ax.scatter(x1.take(ix), x2.take(ix), c=col, marker='o', label=l)
    plt.xlabel(x1_label)
    plt.ylabel(x2_label)
    plt.legend(loc='upper right')
    plt.title(f"Scatter plot of {x1_label} against {x2_label}")
    #fig_name = 'dataset1.png' if first else 'dataset2.png'
    #plt.savefig(fig_name)
    plt.show()

def graph_3D_data(x1, x2, x3, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for yi in np.unique(y):
        ix = np.where(y == yi)[0]
        col = 'blue' if yi else 'lime'
        l = '-1 actual' if yi else '1 actual'
        ax.scatter(x1.take(ix), x2.take(ix), x3.take(ix), c=col, label=l)

    ax.legend();
    ax.set_xlabel('Course Count')
    ax.set_ylabel('Swimming Ability')
    ax.set_zlabel('Sailing Cert')
    plt.title('3D Scatter Plot of Data')

def KFolds_polynomial_features(dataset):
    # Using sklearn augment the two features in the dataset with polynomial features
    # and train a Logistic Regression classifier with L2 penalty added to the cost function.
    course_counts      = dataset.iloc[:,0]
    swimming_abilities = dataset.iloc[:,1]
    X = np.column_stack((course_counts, swimming_abilities))
    y = dataset.iloc[:,2]
    # Steps: 1. For a number of polynomial values [1, 2, 3, 4, 5, 6]:
    #        2. Generate the polynomial data and the model.
    #        3. Do k-fold cross-validation for some k value, like 5.
    #        4. Train the model and get the F1-score.
    #        5. Choosing q too small or too large increases the prediction error on the test data but not on the training data.
    k_fold = KFold(n_splits=5)
    q_range = [1, 2, 3, 4, 5, 6]
    q_mean_f1 = []
    q_std_error = []
    for q in q_range:
        x_poly = PolynomialFeatures(q).fit_transform(X)
        # NOTE: I choose l2 becaues I never want the model to fully zero out any inputs.
        model = LogisticRegression(penalty='l2', solver='lbfgs')
        f1_score_buffer = []
        for train, test in k_fold.split(x_poly):
            model.fit(x_poly[train], y[train])
            y_pred = model.predict(x_poly[test])
            score = f1_score(y_true=y[test], y_pred=y_pred)
            f1_score_buffer.append(score)
        q_mean_f1.append(np.array(f1_score_buffer).mean())
        q_std_error.append(np.array(f1_score_buffer).std())

    plt.errorbar(q_range, q_mean_f1, yerr=q_std_error, linewidth=3)
    plt.xlabel('q')
    plt.title('Plot of maximum polynomial features(q), vs. mean F1-score of the model')
    plt.ylabel('Mean F1 score')
    #fig_name = 'kfolds_poly_i.png' if first else 'kfolds_poly_ii.png'
    #plt.savefig(fig_name)
    #plt.clf()
    plt.show()

def KFolds_C_penalty(dataset):
    # Steps: 1. For a number of C values [0.05, 0.1, 0.5, 1, 5, 10, 50] generate a model.
    #        2. Do k-fold cross validaiton and fit the model to get the F1-score.
    #        2. Plot distribution of prediction error.
    #        3. Choose the lowest C value that stil gives good results.
    # When doing cross-validation, k-folds, we want the smallest C value that still produces good results.
    # This is because we want our model to be as simple as possible so that it is not over-fitted on the training data set.
    course_counts      = dataset.iloc[:,0]
    swimming_abilities = dataset.iloc[:,1]
    X = np.column_stack((course_counts, swimming_abilities))
    y = dataset.iloc[:,2]
    c_range = [0.001, 0.05, 0.1, 0.5, 1, 5, 10, 20]
    c_mean_f1 = []
    c_std_error = []
    for c in c_range:
        x_poly = PolynomialFeatures(1).fit_transform(X)
        model = LogisticRegression(penalty='l2', solver='lbfgs', C=c)
        scores = cross_val_score(model, x_poly, y, cv=5, scoring='f1')
        c_mean_f1.append(np.array(scores).mean())
        c_std_error.append(np.array(scores).std())

    plt.errorbar(c_range, c_mean_f1, yerr=c_std_error, linewidth=3)
    plt.title('Plot of c hyperparameter vs. mean F1-score of the model')
    plt.xlabel('c')
    plt.ylabel('Mean F1 score')
    #fig_name = 'kfolds_c_i.png' if first else 'kfold_c_ii.png'
    #plt.savefig(fig_name)
    #plt.clf()
    plt.show()

def generate_confusion_matrix(dataset):
    best_q = 1
    best_c = 0.5

    train_set, test_set = train_test_split(dataset, test_size=0.2)
    x_train = np.column_stack((train_set.iloc[:,0], train_set.iloc[:,1]))
    y_train = train_set.iloc[:,2]
    x_test  = np.column_stack((test_set.iloc[:,0], test_set.iloc[:,1]))
    y_test  = test_set.iloc[:,2]

    # Calculate the confusion matrices for your trained Logistic Regression
    x_poly_train = PolynomialFeatures(best_q).fit_transform(x_train)
    x_poly_test  = PolynomialFeatures(best_q).fit_transform(x_test)
    model = LogisticRegression(penalty='l2', solver='lbfgs', C=best_c)
    model.fit(x_poly_train, y_train)
    y_pred = model.predict(x_poly_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Confustion Matrix: Logistic Regression")
    print("[TP, FP]: [", tp, ", ", fp, "]")
    print("[FN, TN]: [", fn, ", ", tn, "]")

def compare_most_frequent(dataset):
    train_set, test_set = train_test_split(dataset, test_size=0.2)
    x_train = np.column_stack((train_set.iloc[:,0], train_set.iloc[:,1]))
    y_train = train_set.iloc[:,2]
    x_test  = np.column_stack((test_set.iloc[:,0], test_set.iloc[:,1]))
    y_test  = test_set.iloc[:,2]

    # Also calculate the confusion matrix for one ore more baseline classifier.
    dummy = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    y_pred = dummy.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("Confustion Matrix: Dummy(Most Frequent)")
    print("[TP, FP]: [", tp, ", ", fp, "]")
    print("[FN, TN]: [", fn, ", ", tn, "]")
    print()

    
def ROC_curve(dataset):
    best_q = 1
    best_c = 0.5
    # Plot the ROC curves for your trained Logistic Regression and kNN classifiers.
    # Also plot the point(s) on the ROC plot corresponding to the baseline classifiers.
    # Be sure to include enough points in the ROC curves to allow the detailed shape to be seen.
    train_set, test_set = train_test_split(dataset, test_size=0.2)
    x_train = np.column_stack((train_set.iloc[:,0], train_set.iloc[:,1]))
    y_train = train_set.iloc[:,2]
    x_test  = np.column_stack((test_set.iloc[:,0], test_set.iloc[:,1]))
    y_test  = test_set.iloc[:,2]

    # Calculate the confusion matrices for your trained Logistic Regression
    x_poly_train = PolynomialFeatures(best_q).fit_transform(x_train)
    x_poly_test  = PolynomialFeatures(best_q).fit_transform(x_test)
    model = LogisticRegression(penalty='l2', solver='lbfgs', C=best_c)
    model.fit(x_poly_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, model.decision_function(x_poly_test))
    plt.plot(fpr, tpr)
    plt.title('Plot of ROC curve for the trained Logistic Regression model')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot([0,1], [0,1], color='green', linestyle='--')
    #fig_name = 'roc_logistic_i.png' if first else 'roc_logistic_ii.png'
    #plt.savefig(fig_name)
    #plt.clf()
    plt.show()



# ----------------------------------------------- Plot Feature Visualisation -----------------------------------------------
def normaliseData(X):
    shift = np.average(X)
    scalingFactor = np.max(X) - np.min(X)
    X = (X-shift) / scalingFactor
    return X

def displayOriginalData(dataset):
    fig = plt.figure()
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    course_counts = dataset.iloc[:,0]
    swimming_abilities = dataset.iloc[:,1]
    sailing_certs = dataset.iloc[:,2]

    course_counts = normaliseData(course_counts)
    swimming_abilities = normaliseData(swimming_abilities)
    sailing_certs = normaliseData(sailing_certs)
    y = dataset.iloc[:,3]
    for yi in np.unique(y):
        ix = np.where(y == yi)[0]
        col = 'blue' if yi else 'lime'
        l = 'student' if yi else 'member'

        ax1.scatter(course_counts.take(ix), sailing_certs.take(ix), c=col, label=l)
        ax2.scatter(course_counts.take(ix), swimming_abilities.take(ix), c = col, label = l)
        ax3.scatter(sailing_certs.take(ix), swimming_abilities.take(ix), c = col, label = l)

    ax3.legend()
    
    ax1.set_xlabel('Course Count')
    ax1.set_ylabel('Sailing Certs')
    ax2.set_title('Scatter Plots of Normalised Data')
    ax2.set_xlabel('Course Count')
    ax2.set_ylabel('Swimming Ability')
    ax3.set_xlabel('Sailing Certs')
    ax3.set_ylabel('Swimming Ability')
    plt.tight_layout()
    plt.show()


def course_count_data(r):
    data = []
    for _ in range(r):
        gen = np.arange(-1, 1, 1 / 4)
        for g in gen:
            data.append(g)
    return data

def ability_data(r, c):
    data = []
    gen = np.arange(-1, 1, 1 / 3)
    for i in range(r):
        for _ in range(c):
            data.append(gen[i])
    return data


if __name__ == "__main__":
    csv_files = load_csv_files()
    #data = generate_data("2020", csv_files)

    is_member = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                          1, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1])

    course_count_data = np.array(course_count_data(8))
    ability_data = np.array(ability_data(6, 8))

    graph_2D_data(course_count_data, ability_data, is_member, 'Course Count', 'Swimming Ability')

    dataset = data_to_numpy_dataset(course_count_data, ability_data, is_member)

    KFolds_polynomial_features(dataset)
    KFolds_C_penalty(dataset)
    generate_confusion_matrix(dataset)
    compare_most_frequent(dataset)
    ROC_curve(dataset)

