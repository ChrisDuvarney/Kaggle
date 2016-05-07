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
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB



def writeToCSV(answer):
    with open('SVCsubmit.csv', 'wb') as csvfile:
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
    clf = AdaBoostClassifier()
    classifiers = [
        #SVC(probability=True),
        DecisionTreeClassifier(max_depth=200, criterion="entropy",)]
        #KNeighborsClassifier()]
        #SGDClassifier(loss="log"),
        #LinearDiscriminantAnalysis(),]
        #QuadraticDiscriminantAnalysis()]
    '''{"C":[1.0,2.0,4.0, .5, .3], "kernel":['rbf','linear', 'sigmoid', 'poly'], 'gamma':[.001, .01, .1, 1, 10], 
    'shrinking':[True,False], 'cache_size':[600], 'verbose':[False], 'random_state':[5]},'''
    param_dist = {"base_estimator":classifiers, "n_estimators":[525,550,575], "learning_rate":[1.45,1.5,1.55], "algorithm":['SAMME.R',], "random_state":[None]}
    #ftwo_scorer = roc_auc_score()
    random_search = GridSearchCV(clf, param_grid=param_dist, cv=3, n_jobs=3, verbose=50, scoring='roc_auc')
    random_search.fit(vals,actions)
    joblib.dump(random_search, 'SVCsearch.pkl')
    random_search = joblib.load('SVCsearch.pkl')
    print(random_search.best_estimator_)
    #print(random_search.grid_scores_)
    print(random_search.best_score_)
    writeToCSV(random_search.predict_proba(totalTest)[:,1])
    print(random_search.score(X_test,y_test))

if __name__ == '__main__':
    main()