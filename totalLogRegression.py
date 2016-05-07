from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import train_test_split
import numpy
import pickle
import csv
from scipy import stats
from scipy.stats import randint as sp_randint
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import fbeta_score, make_scorer,roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB



def writeToCSV(answer):
    with open('regreg.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',)
        spamwriter.writerow(['Id','Action'])
        for item in range(len(answer)):
            spamwriter.writerow([item+1,answer[item]])

def getPercent(guess,correct):
    totalMiss = 0
    for i in range(len(guess)):
        if guess[i]!=correct[i]:
            #print(guess[i], correct[i])
            totalMiss+=1


    return (float(totalMiss)/len(guess))


def matrixFromCSV(path):
    csvFile = open(path).read().split("\n")
    vals = csvFile[1:]
    listOfVals = []
    listOfActions = []
    highest = [312153, 311696, 311178, 286791, 286792, 311867, 311867, 308574, 270691]
    for elem in vals:
        stringVals = elem.split(",")[1:]
        intVals = numpy.array([float(stringVals[i])/highest[i] for i in range(len(stringVals))])
        #intVals.reshape(9,1)
        if len(intVals)!=0:
           listOfVals.append(intVals[:-2])
        try:
            listOfActions.append(int(elem.split(",")[0]))
        except:
            pass
    return numpy.array(listOfVals), numpy.array(listOfActions)  

def main():
    vals, actions = matrixFromCSV("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\train.csv")
    X_train, X_test, y_train, y_test = train_test_split(vals, actions, test_size=0.33, random_state=22)
    totalTest, totalAns = matrixFromCSV("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\test.csv")
    testFeatures = []
    csvFile1 = open("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\GradientBoostinglog.csv").read().split("\n")
    csvFile2 = open("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\adaBoostlog.csv").read().split("\n")
    csvFile3 = open("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\randomForestlog.csv").read().split("\n")
    csvFile4 = open("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\logistic_regression_predLog.csv").read().split("\n")
    features = []
    for elem in range(1,len(csvFile1)):
        if len(csvFile3[elem])!=0:
            elem1 = float(csvFile1[elem].split(',')[1])
            elem2 = float(csvFile2[elem].split(',')[1])
            elem3 = float(csvFile3[elem].split(',')[1])
            elem4 = float(csvFile4[elem].split(',')[1])
            temp = [elem1, elem3,elem4]
            features.append(temp)

    clf = LogisticRegression()
    param_dist = [
      {'penalty':['l2'], 'dual':[False], 'tol':[0.0001, ], 'C':[1.35,1.4, ], 'fit_intercept':[True], 
      'intercept_scaling':[1,], 'class_weight':[None], 'random_state':[10], 
      'solver':["lbfgs",], 'max_iter':[50, 100,200,30], 'multi_class':['multinomial'], 'verbose':[0], 
      'warm_start':[False], 'n_jobs':[1]}]
    random_search = GridSearchCV(clf, param_grid=param_dist,
        cv=5, n_jobs=3, verbose=50, scoring='roc_auc')
    random_search.fit(features,y_test)
    csvFile1 = open("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\GradientBoosting.csv").read().split("\n")
    csvFile2 = open("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\adaBoost.csv").read().split("\n")
    csvFile3 = open("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\randomForest.csv").read().split("\n")
    csvFile4 = open("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\logistic_regression_pred.csv").read().split("\n")
    finalPred = []
    for elem in range(1,len(csvFile1)):
        if len(csvFile3[elem])!=0:
            elem1 = float(csvFile1[elem].split(',')[1])
            elem2 = float(csvFile2[elem].split(',')[1])
            elem3 = float(csvFile3[elem].split(',')[1])
            elem4 = float(csvFile4[elem].split(',')[1])
            temp = [elem1, elem3,elem4]
            finalPred.append(temp)
    #print(finalPred)
    print(random_search.best_estimator_)
    #print(random_search.grid_scores_)
    print(random_search.best_score_)
    writeToCSV(random_search.predict_proba(finalPred)[:,1])

if __name__ == '__main__':
    main()