from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

import mlflow

if __name__ == '__main__':
    
    # load in dataset
    cancer = datasets.load_breast_cancer(as_frame=True)

    # train, test, split
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)
    
    # call classifier, fit, and run prediction
    clf = svm.SVC(C=2, kernel="linear")
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    #print(clf.get_params())

    # identify accuracy
    print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))