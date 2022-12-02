
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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Lasso, Ridge

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
    global features, targets
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
    
    features = np.column_stack((course_level_type, course_startDate))
    targets = np.array(course_numOfMembers)


# ----------------------------------------------- 3D Scatter Plot Data -----------------------------------------------
def threeDScatterPlot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(features[:, 0], features[:, 1], targets)
    ax.set_xlabel('course_type.course_level')
    ax.set_ylabel('course_startDate')
    ax.set_zlabel('number_of_members')

    plt.title('3D Scatter Plot of Data')
    plt.show()


# ----------------------------------------------- Lasso Regression -----------------------------------------------
# Creates Lasso Regression Models
def trainLassoRegressionModel():
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20)
    polynomial_deg = PolynomialFeatures(degree=5)

    # new feature matrix of features up to power 5
    polynomial_features = polynomial_deg.fit_transform(X_train)

    c_arr = [1, 10, 1000]
    for c in c_arr:
        penalty = 1/(2*c)
        lasso_model = Lasso(alpha=penalty)
        lasso_model.fit(polynomial_features, y_train)
        print('Lasso Regression, C=', c)
        print('Coefficients : ', lasso_model.coef_)
        print('Intercept : ', lasso_model.intercept_, '\n')

        title_lasso = 'Lasso Test Results, C=' + str(c)

        # Send Lasso Model to plot function for each value of C
        plotRegressionModel(lasso_model, polynomial_deg, title_lasso)

# Preform kFold for 5 splits for Lasso Regression
# Vary C value (1, 5, 10, 15, 20, 25, 50, 100)
# Plot graph to analyse the data
def perform5FoldCrossValidationLasso():
    polynomial_deg = PolynomialFeatures(degree=5)
    standard_devs = []
    means = []
    c_arr = [1, 5, 10, 15, 20, 25, 50, 100]
    for c in c_arr:
        penalty = 1 / (2 * c)
        lasso_model = Lasso(alpha=penalty)
        five_fold = KFold(n_splits=5, shuffle=False)
        estimates_mean_sq_err = []

        for train_index, test_index in five_fold.split(targets):
            X_train, X_test, y_train, y_test = features[train_index], features[test_index], targets[train_index], targets[test_index]
            polynomial_features = polynomial_deg.fit_transform(X_train)
            lasso_model.fit(polynomial_features, y_train)
            polynomial_Xtest = polynomial_deg.fit_transform(X_test)
            y_pred = lasso_model.predict(polynomial_Xtest)
            estimates_mean_sq_err.append(mean_squared_error(y_test, y_pred))

        means.append(np.mean(estimates_mean_sq_err))
        standard_devs.append(np.std(estimates_mean_sq_err))

    plt.errorbar(c_arr, means, yerr=standard_devs, ecolor='r', linewidth=2, capsize=5)
    plt.title('5 Fold Lasso with various C values')
    plt.xlabel('C')
    plt.ylabel('Mean (blue) / Standard Deviation (red)')
    plt.show()


# ----------------------------------------------- Ridge Regression -----------------------------------------------
# Same as above but with Ridge Regression rather than Lasso Regression
# Creates Ridge Regression Models
def trainRidgeRegressionModel():
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20)
    polynomial_deg = PolynomialFeatures(degree=5)
    polynomial_features = polynomial_deg.fit_transform(X_train)

    c_arr = [1, 10, 1000]
    for c in c_arr:
        penalty = 1 / (2 * c)
        ridge_model = Ridge(alpha=penalty)
        ridge_model.fit(polynomial_features, y_train)
        print('Ridge Regression, C=', c)
        print('Coefficients : ', ridge_model.coef_)
        print('Intercept : ', ridge_model.intercept_, '\n')

        title_ridge = 'Ridge Test Results, C=' + str(c)

        # Send Lasso Model to plot function for each value of C
        plotRegressionModel(ridge_model, polynomial_deg, title_ridge)

# Preform kFold for 5 splits for Ridge Regression
# Vary C value (0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10)
# Plot graph to analyse the data
def perform5FoldCrossValidationRidge():
    polynomial_deg = PolynomialFeatures(degree=5)
    standard_devs = []
    means = []
    c_arr = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    for c in c_arr:
        penalty = 1 / (2 * c)
        ridge_model = Ridge(alpha=penalty)
        five_fold = KFold(n_splits=5, shuffle=False)
        estimates_mean_sq_err = []

        for train_index, test_index in five_fold.split(features):
            X_train, X_test, y_train, y_test = features[train_index], features[test_index], targets[train_index], targets[test_index]
            polynomial_features = polynomial_deg.fit_transform(X_train)
            ridge_model.fit(polynomial_features, y_train)
            polynomial_Xtest = polynomial_deg.fit_transform(X_test)
            y_pred = ridge_model.predict(polynomial_Xtest)
            estimates_mean_sq_err.append(mean_squared_error(y_test, y_pred))

        means.append(np.mean(estimates_mean_sq_err))
        standard_devs.append(np.std(estimates_mean_sq_err))

    plt.errorbar(np.log10(c_arr), means, yerr=standard_devs, ecolor='r', linewidth=2, capsize=5)
    plt.title('5 Fold Ridge with various C values')
    plt.xlabel('log10(C)');
    plt.ylabel('Mean (blue) / Standard Deviation (red)')
    plt.show()


# ----------------------------------------------- Regression Model Helper Functions -----------------------------------------------
# Plot the Regression Model
def plotRegressionModel(lasso_model, polynomial_deg, title):
    Xtest = test_space()

    polynomial_Xtest = polynomial_deg.fit_transform(Xtest)
    y_pred = lasso_model.predict(polynomial_Xtest)

    graph_surface(y_pred, title)

# Create test space
def test_space():
    Xtest = []
    # 6 on Xaxis as there is no course type greater than 6
    # 4 and 12 on Yaxis as the min month numneber of a course is 4 and the max month number is 12
    Xgrid = np.linspace(0, 6)
    Ygrid = np.linspace(4, 12)
    for i in Xgrid:
        for j in Ygrid:
            Xtest.append([i, j])
    Xtest = np.array(Xtest)
    return Xtest

# Graph scatter and surface on the same plot
def graph_surface(y_pred, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Xtest_graph = test_space()
    surface = ax.plot_trisurf(Xtest_graph[:, 0], Xtest_graph[:, 1], y_pred, cmap='viridis')
    ax.scatter(features[:, 0], features[:, 1], targets, label='Training data')
    ax.set_xlabel('course_type.course_level')
    ax.set_ylabel('course_startDate')
    ax.set_zlabel('number_of_members')
    fig.colorbar(surface, label='Predictions', shrink=0.5, aspect=8)
    plt.title(title)
    plt.legend()
    plt.show()


# ----------------------------------------------- Main -----------------------------------------------
if __name__ == "__main__":

    # Extract Data
    csv_files = load_csv_files()
    courses = extractData(csv_files)

    # Convert courses to seperate arrays
    convertToSeperateArrays(courses)

    # 3D Scatter Plot of the data
    #threeDScatterPlot()

    # Lasso Regression
    #trainLassoRegressionModel()
    perform5FoldCrossValidationLasso()

    # Ridge Regression
    #trainRidgeRegressionModel()
    perform5FoldCrossValidationRidge()