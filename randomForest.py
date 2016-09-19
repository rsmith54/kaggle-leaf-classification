import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
#from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

def exportDataFrame(filename, showHead = False):
    # For .read_csv, always use header=0 when you know row 0 is the header row
    df = pd.read_csv(filename , header=0)

    if showHead : print(df.head(10))
    print(df.info())
    return df

def main():
    train_data = exportDataFrame("csv/train.csv", showHead = True)
    test_data = exportDataFrame("csv/test.csv")

    test_ids = test_data.pop('id')
    columns = np.unique(train_data['species'].values)

    train = {'x' : train_data.ix[:, 2:] ,
             'Y' : train_data.ix[:, 1 ] ,
    }

    cl = RandomForestClassifier(n_estimators=100, n_jobs=4)
    cl.fit(train['x'] , train['Y'])
    prediction_proba = cl.predict_proba(test_data.ix[:, :])

    submission = pd.DataFrame(prediction_proba, index=test_ids, columns=columns)
#    print(submission)
    submission.to_csv('submission_randomForest.csv')

if __name__ == '__main__':
    main()
