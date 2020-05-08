import os

''' CHECK OS - following line to be executed only on Windows machines'''
if(os.name == "nt"):
    os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))



###########################################################################
###########################################################################
'''                 IMPORT CN PROTECT MODULES                           '''
###########################################################################
###########################################################################
from cn.protect import Protect
from cn.protect.privacy import KAnonymity
from cn.protect.hierarchy import DataHierarchy, OrderHierarchy
from cn.protect.quality import Loss


###########################################################################
###########################################################################
'''                 IMPORT SKLEARN MODULES                              '''
###########################################################################
###########################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


###########################################################################
###########################################################################
'''                 IMPORT OTHER STATS MODULES                          '''
###########################################################################
###########################################################################
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.naive_bayes import GaussianNB


###########################################################################
###########################################################################
'''               DEFINE MODULES TO TRAIN ML MODELS                     '''
###########################################################################
###########################################################################


''' Logistic Regression '''

def log_reg(X_train, Y_train, X_test, Y_test):

    log_reg_clf = LogisticRegression(random_state=0, penalty='l1', solver='liblinear', max_iter=100)
    log_reg_clf.fit(X_train, Y_train)
    return(accuracy_score(Y_test, log_reg_clf.predict(X_test)))


''' Decision Tree '''
def dec_tree(X_train, Y_train, X_test, Y_test):

    dec_tree_clf = DecisionTreeClassifier(random_state = 42,max_depth=5)
    dec_tree_clf.fit(X_train, Y_train)
    return accuracy_score(Y_test, dec_tree_clf.predict(X_test))


''' K nearest neighbors classifier '''
def knn_clf(X_train, Y_train, X_test, Y_test):

    knn_clf = KNeighborsClassifier(n_neighbors=6, algorithm='ball_tree')
    knn_clf.fit(X_train, Y_train)
    return accuracy_score(Y_test, knn_clf.predict(X_test))


''' Support Vector Machine '''
def SVM(X_train, Y_train, X_test, Y_test):

    sup_vec_clf = SVC(gamma='auto')
    sup_vec_clf.fit(X_train, Y_train)
    return accuracy_score(Y_test, sup_vec_clf.predict(X_test))


'''Define global function to make calls to each model'''
def test_models(X_train, Y_train, X_test, Y_test):

    #CALL each model
    acc = []
    acc.append(log_reg(X_train, Y_train, X_test, Y_test))
    acc.append(dec_tree(X_train, Y_train, X_test, Y_test))
    acc.append(knn_clf(X_train, Y_train, X_test, Y_test))
    acc.append(SVM(X_train, Y_train, X_test, Y_test))

    return acc

'''test the ML models on UNPROTECTED dataset and append results'''
def unprotected_data_results():

    unprotected_data_copy = adult_data.copy(deep=False).drop('ssn',axis=1)
    unprotected_data_copy['salary_class'] = unprotected_data_copy['salary_class'].astype('category').cat.codes
    unprotected_model_data = pd.get_dummies(unprotected_data_copy, columns=[col for col in adult_data if col not in ('ssn','salary_class')])
    X_train_unprotected, X_test_unprotected, Y_train_unprotected, Y_test_unprotected = train_test_split(unprotected_model_data.drop('salary_class',axis=1), unprotected_data_copy['salary_class'], test_size=0.2)
    acc = test_models(X_train_unprotected, Y_train_unprotected, X_test_unprotected, Y_test_unprotected)

    return acc



'''test the ML models on PROTECTED dataset and append results'''
def protected_data_results(protected_model_data, protected_data_copy):

    X_train_protected, X_test_protected, Y_train_protected, Y_test_protected = train_test_split(protected_model_data.drop('salary_class',axis=1), protected_data_copy['salary_class'], test_size=0.2)
    if(len(np.unique(Y_train_protected)) == 1):
        print("All one class! Please provide lower anonymity value")
        # continue
    acc = test_models(X_train_protected, Y_train_protected, X_test_protected, Y_test_protected)

    return acc

def get_results():

    #accuracies dictionary
    acc = {}
    acc["Unprot_SVM"] = list()
    acc["Unprot_Logistic"] = list()
    acc["Unprot_Decision Tree"] = list()
    acc["Unprot_KNeighbors"] = list()
    acc["Prot_SVM"] = list()
    acc["Prot_Logistic"] = list()
    acc["Prot_Decision Tree"] = list()
    acc["Prot_KNeighbors"] = list()

    #storing loss and risk values for unprotected data
    loss_values = list()
    risk_values = list()

    #get accuracies for unprotected data
    accuracies = unprotected_data_results()
    acc["Unprot_Logistic"].append(accuracies[0])
    acc["Unprot_Decision Tree"].append(accuracies[1])
    acc["Unprot_KNeighbors"].append(accuracies[2])
    acc["Unprot_SVM"].append(accuracies[3])
    # get accuracies for protected data
    anonymity_levels = [45, 300, 500, 1500]
    for k in anonymity_levels:
        print("Applying k = ",k)
        priv, loss, risk = anonymize(adult_data, k)
        loss_values.append(loss)
        risk_values.append(risk)

        protected_data_copy = priv.copy(deep=False).drop('ssn',axis=1)
        protected_data_copy['salary_class'] = protected_data_copy['salary_class'].astype('category').cat.codes
        protected_model_data = pd.get_dummies(protected_data_copy, columns=[col for col in priv if col not in ('ssn','salary_class')])
        accuracies = protected_data_results(protected_model_data, protected_data_copy)
        acc["Prot_Logistic"].append(accuracies[0])
        acc["Prot_Decision Tree"].append(accuracies[1])
        acc["Prot_KNeighbors"].append(accuracies[2])
        acc["Prot_SVM"].append(accuracies[3])

    return anonymity_levels, acc

''' Function to plot output '''

def plot(anonymity_levels, acc):

    plot1 = plt.figure(1)
    plt.plot(anonymity_levels, acc["Unprot_Decision Tree"]*4, color = 'blue', label='Decision Tree')
    plt.plot(anonymity_levels, acc["Unprot_Logistic"]*4,color = 'red', label='Logistic Regression')
    plt.plot(anonymity_levels, acc["Unprot_SVM"]*4, color = 'green', label='SVM')
    plt.plot(anonymity_levels, acc["Unprot_KNeighbors"]*4, color = 'orange', label='KNeighbors')
    plt.xlabel("")
    plt.ylabel("Accuracy of classifiers before protection")
    plt.legend()
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)

    plot2 = plt.figure(2)
    plt.plot(anonymity_levels, acc["Prot_Logistic"], color = 'red', label='Logistic Regression')
    plt.plot(anonymity_levels, acc["Prot_Decision Tree"], color = 'blue', label='Decision Tree')
    plt.plot(anonymity_levels, acc["Prot_SVM"], color = 'green', label='SVM')
    plt.plot(anonymity_levels, acc["Prot_KNeighbors"], color = 'orange', label='KNeighbors')
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy of classifiers after protection")
    plt.legend()
    plt.show()


###########################################################################
###########################################################################
'''                       PREPROCESSING DATA                            '''
###########################################################################
###########################################################################

def create_education_hierarchy(adult_data):
    education_data_original = list(adult_data["education"].unique())
    education_data_level1 = []
    education_data_level2 = []
    education_data_level3 = []
    for i in education_data_original:
        if i in ['Preschool','1st-4th','5th-6th']:
            education_data_level1.append('Primary School')
            education_data_level2.append("Primary Education")
        elif i in ['7th-8th','9th','10th','11th','12th','HS-grad']:
            education_data_level1.append('High School')
            education_data_level2.append("Secondary Education")
        elif i in ['Bachelors','Some-college']:
            education_data_level1.append('Undergraduate')
            education_data_level2.append("Higher Education")
        elif i in ['Masters','Doctorate']:
            education_data_level1.append('Graduate')
            education_data_level2.append("Higher Education")
        else:
            education_data_level1.append('Professional Education')
            education_data_level2.append("Higher Education")
        education_data_level3.append("*")
    education_hierarchy = pd.DataFrame({"Original":education_data_original,"Level1":education_data_level1,"Level2":education_data_level2,"Level3":education_data_level3})
    education_hierarchy.to_csv("adult_hierarchy_education.csv",index=False)


def create_native_country_hierarchy(adult_data):
    native_country_data_original = list(adult_data["native_country"].unique())
    native_country_data_level1 = []
    native_country_data_level2 = []
    native_country_data_level3 = []
    for i in native_country_data_original:
        if i == "South-Africa":
            native_country_data_level1.append("Africa")
            native_country_data_level2.append("Middle")
        elif i in ["Cambodia","China","Hong-Kong","India","Iran","Japan","Laos","Philippines","Taiwan","Thailand","Vietnam"]:
            native_country_data_level1.append("Asia")
            native_country_data_level2.append("East")
        elif i in ["England","France","Germany","Greece","Holland","Hungary","Ireland","Italy","Poland","Portugal","Scotland","Yugoslavia"]:
            native_country_data_level1.append("Europe")
            native_country_data_level2.append("Middle")
        elif i in ["Colombia","Ecuador","Nicaragua","Peru","Trinidad&Tobago"]:
            native_country_data_level1.append("South America")
            native_country_data_level2.append("West")
        else:
            native_country_data_level1.append("North America")
            native_country_data_level2.append("West")
        native_country_data_level3.append("*")
    native_country_hierarchy = pd.DataFrame({"Original":native_country_data_original,"Level1":native_country_data_level1,"Level2":native_country_data_level2,"Level3":native_country_data_level3})
    native_country_hierarchy.to_csv("adult_hierarchy_native_country.csv",index=False)


def create_marital_status_hierarchy(adult_data):
    marital_status_data_original = list(adult_data["marital_status"].unique())
    marital_status_data_level1 = []
    marital_status_data_level2 = []
    for i in marital_status_data_original:
        if i in ["Married-civ-spouse","Married-AF-spouse"]:
            marital_status_data_level1.append("Spouse present")
        else:
            marital_status_data_level1.append("Spouse not present")
        marital_status_data_level2.append("*")
    marital_status_hierarchy = pd.DataFrame({"Original":marital_status_data_original,"Level1":marital_status_data_level1,"Level2":marital_status_data_level2})
    marital_status_hierarchy.to_csv("adult_hierarchy_marital_status.csv",index=False)


def create_race_hierarchy(adult_data):
    race_data_original = list(adult_data["race"].unique())
    race_data_level1 = []
    for i in race_data_original:
        race_data_level1.append("*")
    race_hierarchy = pd.DataFrame({"Original":race_data_original,"Level1":race_data_level1})
    race_hierarchy.to_csv("adult_hierarchy_race.csv",index=False)


def create_sex_hierarchy(adult_data):
    sex_data_original = list(adult_data["sex"].unique())
    sex_data_level1 = []
    for i in sex_data_original:
        sex_data_level1.append("*")
    sex_hierarchy = pd.DataFrame({"Original":sex_data_original,"Level1":sex_data_level1})
    sex_hierarchy.to_csv("adult_hierarchy_sex.csv",index=False)


def create_salary_class_hierarchy(adult_data):
    salary_class_data_original = list(adult_data["salary_class"].unique())
    salary_class_data_level1 = []
    for i in salary_class_data_original:
        salary_class_data_level1.append("*")
    salary_class_hierarchy = pd.DataFrame({"Original":salary_class_data_original,"Level1":salary_class_data_level1})
    salary_class_hierarchy.to_csv("adult_hierarchy_salary_class.csv",index=False)


def create_workclass_hierarchy(adult_data):
    workclass_data_original = list(adult_data["workclass"].unique())
    workclass_data_level1 = []
    workclass_data_level2 = []
    for i in workclass_data_original:
        if i in ["Private","Self-emp-not-inc","Self-emp-inc"]:
            workclass_data_level1.append("Non-government")
        elif i in ["Federal-gov","Local-gov","State-gov"]:
            workclass_data_level1.append("Government")
        else:
            workclass_data_level1.append("Unemployed")
        workclass_data_level2.append("*")
    workclass_hierarchy = pd.DataFrame({"Original":workclass_data_original,"Level1":workclass_data_level1,"Level2":workclass_data_level2})
    workclass_hierarchy.to_csv("adult_hierarchy_workclass.csv",index=False)


def create_occupation_hierarchy(adult_data):
    occupation_data_original = list(adult_data["occupation"].unique())
    occupation_data_level1 = []
    occupation_data_level2 = []
    for i in occupation_data_original:
        if i in ["Exec-managerial","Handlers-cleaners","Sales"]:
            occupation_data_level1.append("Non-technical")
        elif i in ["Craft-repair","Machine-op-inspct","Prof-specialty","Tech-support"]:
            occupation_data_level1.append("Technical")
        else:
            occupation_data_level1.append("Other")
        occupation_data_level2.append("*")
    occupation_hierarchy = pd.DataFrame({"Original":occupation_data_original,"Level1":occupation_data_level1,"Level2":occupation_data_level2})
    occupation_hierarchy.to_csv("adult_hierarchy_occupation.csv",index=False)


def anonymize(adult_data, k):
    prot = Protect(adult_data, KAnonymity(k))

    prot.quality_model = Loss()

    prot.suppression = 0.1

    prot.itypes['ssn'] = 'identifying'
    prot.itypes['salary_class'] = 'insensitive'
    for col in adult_data.columns:
        if col not in ('ssn','salary_class'):
            prot.itypes[col] = 'quasi'

    prot.hierarchies["age"] = OrderHierarchy('interval', 5, 2, 2)
    for col in adult_data.columns:
        if col not in ("age","ssn"):
            prot.hierarchies[col] = DataHierarchy(pd.read_csv("./adult_hierarchy_"+col+".csv"))

    priv = prot.protect()

    return priv, prot.stats.informationLoss, prot.stats.highestRisk


###########################################################################
###########################################################################
'''                      BEGIN MAIN FUNCTION                            '''
###########################################################################
###########################################################################

if(__name__ == "__main__"):

    adult_data = pd.read_csv("./adult.csv")
    for i in adult_data:
        if i not in ('age','ssn'):
            hierarchy_func = globals()["create_"+i+"_hierarchy"]
            hierarchy_func(adult_data)


    #Call function to execute anonymization and machine learning models for the dataset and get results.
    anonymity_levels, accuracies = get_results()
    plot(anonymity_levels, accuracies)


    exit("Reached end of the program")
