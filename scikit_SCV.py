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



def writeToCSV(answer):
    with open('submit.csv', 'wb') as csvfile:
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
           listOfVals.append(intVals)
        try:
            listOfActions.append(int(elem.split(",")[0]))
        except:
            pass
    return numpy.array(listOfVals), numpy.array(listOfActions)  

def main():
    vals, actions = matrixFromCSV("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\train.csv")
    X_train, X_test, y_train, y_test = train_test_split(vals, actions, test_size=0.33, random_state=22)
    totalTest, totalAns = matrixFromCSV("C:\\Users\\Chrisd\\Documents\\College\\Spring 2016\\379K\\Kaggle\\Kaggle\\test.csv")
    '''clf = SVC()
    param_dist = {"C":[1.0,2.0,4.0, .5, .3], "kernel":['rbf','linear', 'sigmoid', 'poly'], 'degree':range(2,7), 
    'gamma':[.001, .01, .1, 1, 10], 'shrinking':[True,False], 'tol':[0.001,.0001, .01, .1],
    'cache_size':[600], 'class_weight':[None, 'balanced'], 'verbose':[False], 'decision_function_shape':[None], 'random_state':[5]}

    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
        n_iter=60, cv=5, n_jobs=3)
    random_search.fit(vals,actions)
    joblib.dump(random_search, 'SVCsearch.pkl')'''
    random_search = joblib.load('SVCsearch.pkl')
    writeToCSV(random_search.predict(totalTest))
    print(random_search.score(X_test,y_test))

if __name__ == '__main__':
    main()