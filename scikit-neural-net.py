from sknn.mlp import Classifier, Layer


def matrixFromCSV(path):
    csvFile = open(path).read()
    for elem in csvFile:
        print(elem)

def main():
    matrixFromCSV("C:\\Users\\chrisd\\Documents\\College\\379K\\Kaggle\\train.csv")
    print("here")
    
if __file__ == '__main__':
    main()