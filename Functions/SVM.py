import sklearn
from sklearn import svm
from sklearn import metrics

'''Function performs SVM
Parameters include:
    -Training_data:
    -Training labels: true label for each training vector represented 
                      by one-hot vector y(x) = 0 for honest and y(x) = 1 for malicious
    -Testing_data:
    -Testing_labels:
'''

def eval_SVM(X_train, y_train, X_test, y_test):

    #Define a radial basis function (RBF) Kernel
    svm_mdl = svm.SVC(kernel="rbf")

    #Train the model using the training set
    svm_mdl.fit(X_train,y_train)

    #Predict the response
    y_pred = svm_mdl.predict(X_test)

    #Print out Model Accuracy
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    #Compute DR, FPR, and HD
    DR = metrics.precision_score(y_test, y_pred)
    FPR = 1 - metrics.recall_score(y_test, y_pred, pos_label=0)
    HD = abs(DR - FPR)
    
    return(DR, FPR, HD)