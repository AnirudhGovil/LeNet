# Import required modules
import numpy as np
from RBF import RBF
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

class Classifiers():

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classes = np.unique(self.y_train)
        self.n_classes = len(self.classes)

    def LinearClassifier(self, X_train = [], y_train = []):
        
        if(X_train == [] or y_train == []):
            X_train = self.X_train
            y_train = self.y_train

        # Define the classifier
        clf = LinearSVC(multi_class='crammer_singer')

        # Train the classifier
        clf.fit(X_train, y_train)

        return clf

    def KNN(self):

        # Define the classifier
        clf = KNeighborsClassifier(metric = "euclidean", n_neighbors=3)

        # Train the classifier
        clf.fit(self.X_train, self.y_train)

        return clf

    def PolynomialClassifier(self, X_train, y_train):

        # Polynomial Kernel with degree 2
        clf = svm.SVC(kernel='poly', degree=2, gamma='auto')

        # Train the classifier
        clf.fit(X_train, y_train)

        return clf

    def RBFNetwork(self, X_train = [], y_train = []):
            
            if(X_train == [] or y_train == []):
                X_train = self.X_train
                y_train = self.y_train
    
            # Define the classifier
            clf = RBF(X_train, y_train, self.X_test, self.y_test,num_of_classes=27,
                     k=27, std_from_clusters=False)

            clf.train()

            return clf

    def SVM(self, X_train = [], y_train = []):
        
        if(X_train == [] or y_train == []):
            X_train = self.X_train
            y_train = self.y_train

        # Define the classifier
        clf = svm.SVC(kernel='linear', gamma='auto')

        # Train the classifier
        clf.fit(X_train, y_train)

        return clf




    