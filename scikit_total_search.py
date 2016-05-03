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
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV

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
    classifiers = [
        #KNeighborsClassifier(n_neighbors=10, n_jobs=-1, weights='distance'),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=100, C=.5),
        DecisionTreeClassifier(max_depth=20),
        #RandomForestClassifier(max_depth=20, n_estimators=15, max_features=5),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
    ]
    ans = []
    best = None
    for elem in classifiers:
        elem.fit(X_train,y_train)
        #tester = elem.predict(X_test)
        percent = elem.score(X_test,y_test)
        #percent = getPercent(tester, y_test)
        #print(percent)
        ans.append((str(elem), percent))
        if best==None or percent>best[1]:
            best = (str(elem), percent)

   

    #print(X_test.shape)
    #X_test.reshape(9,1)''
    '''nn = pickle.load(open('nn.pkl', 'rb'))
    answer = nn.predict(X_test)
    writeToCSV(answer)
    print(getPercent(answer,y_test))'''
    print(best)
    
def writeToCSV(answer):
    with open('submit.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',)
        for item in range(len(answer)):
            spamwriter.writerow([item,answer[item][0]])

def getPercent(guess,correct):
    totalMiss = 0
    for i in range(len(guess)):
        if guess[i]!=correct[i]:
            #print(guess[i], correct[i])
            totalMiss+=1


    return (float(totalMiss)/len(guess))

    
main()