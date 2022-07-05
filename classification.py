
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import pickle
from bow_features import FeatureExtraction


class Classifier:


    def read_data(self, train_csv, test_csv):

        df = pd.read_csv(train_csv, sep=',', header=None)
        X_train = df.iloc[:,:-1]
        Y_train = df[df.columns[-1]]

        df = pd.read_csv(test_csv, sep=',', header=None)
        X_test = df.iloc[:, :-1]
        Y_test = df[df.columns[-1]]

        return X_train, Y_train, X_test, Y_test


    def read_test_data(self, test_csv):

        df = pd.read_csv(test_csv, sep=',', header=None)
        X_test = df.iloc[:, :-1]
        Y_test = df[df.columns[-1]]

        return X_test, Y_test


    def read_text(self, txt, label_file):

        fe = FeatureExtraction()
        X, Y = fe.extract_txt_tfidf_feat(txt, label_file)
        return X, Y


    def decision_trees(self, X_train, Y_train, X_test, Y_test, prediction_file, model_file):

        clf = tree.DecisionTreeClassifier(min_samples_leaf=10, random_state=5)
        clf.fit(X_train, Y_train)

        if model_file is not None:
            pickle.dump(clf, open(model_file, 'wb'))

        hyp = clf.predict(X_test)
        self.save_prediction(hyp, prediction_file)
        accuracy = accuracy_score(Y_test, hyp)
        print("Decision Tree: " + str(accuracy))

        precision, recall, f1, support = precision_recall_fscore_support(Y_test, hyp, average='macro')
        print('Precision: ' + str(precision) + ", Recall: " + str(recall) + ", F1: " + str(f1))

    def forest(self, X_train, Y_train, X_test, Y_test, prediction_file, model_file):

        clf = RandomForestClassifier(random_state=3)
        clf.fit(X_train, Y_train)

        if model_file is not None:
            pickle.dump(clf, open(model_file, 'wb'))

        hyp = clf.predict(X_test)
        self.save_prediction(hyp, prediction_file)
        accuracy = accuracy_score(Y_test, hyp)
        print("Random forest: "  + str(accuracy))

        precision, recall, f1, support = precision_recall_fscore_support(Y_test, hyp, average='macro')
        print('Precision: ' + str(precision) + ", Recall: " + str(recall) + ", F1: " + str(f1))

    def svm_ml(self, X_train, Y_train, X_test, Y_test, prediction_file, model_file):

        clf = svm.SVC()
        clf.fit(X_train, Y_train)

        if model_file is not None:
            pickle.dump(clf, open(model_file, 'wb'))

        hyp = clf.predict(X_test)
        self.save_prediction(hyp, prediction_file)
        accuracy = accuracy_score(Y_test, hyp)
        print("SVM: " + str(accuracy))

        precision, recall, f1, support = precision_recall_fscore_support(Y_test, hyp, average='macro')
        print('Precision: ' + str(precision) + ", Recall: " + str(recall) + ", F1: " + str(f1))


    def nb_classifier(self, X_train, Y_train, X_test, Y_test, prediction_file, model_file):

        clf = GaussianNB()
        clf.fit(X_train, Y_train)

        if model_file is not None:
            pickle.dump(clf, open(model_file, 'wb'))

        hyp = clf.predict(X_test)

        self.save_prediction(hyp, prediction_file)
        accuracy = accuracy_score(Y_test, hyp)
        print("Naive Bayes: " + str(accuracy))

        precision, recall, f1, support = precision_recall_fscore_support(Y_test, hyp, average='macro') 
        #precision, recall, f1, support = precision_recall_fscore_support(Y_test, hyp, pos_label=1, zero_division=1, average='macro')
        print('Precision: ' + str(precision) + ", Recall: " + str(recall) + ", F1: " + str(f1))


    def save_prediction(self, hypotheses, prediction_file):
        """
        Save prediction in a file
        :param hypotheses: list, hypothesis
        :param prediction_file: str, text file
        :return: None
        """

        with open(prediction_file, 'w', encoding='utf8') as writer:
            for value in hypotheses:
                writer.write(str(value) + "\n")

    def evaluation(self, X_test, Y_test, model_file, prediction_file):

        # load the model from disk
        clf = pickle.load(open(model_file, 'rb'))

        hyp = clf.predict(X_test)
        self.save_prediction(hyp, prediction_file)
        print(hyp)
        print(Y_test)
        accuracy = accuracy_score(Y_test, hyp)
        print("Accuracy: " + str(accuracy))

        precision, recall, f1, support = precision_recall_fscore_support(Y_test, hyp, pos_label=1, zero_division=1,
                                                                         average='macro')
        print('Precision: ' + str(precision) + ", Recall: " + str(recall) + ", F1: " + str(f1))



def main():

    text = 'edge-text.txt'
    label = 'edge-label.txt'
    result_file = 'result.hyp'
    model = 'classification_model'

    cl = Classifier()
    print('extract data...')
    X, Y = cl.read_text(text, label)

    # Split the data for training, testing and validation
    pos1 = round(len(X) * 0.7)
    pos2 = pos1 + round(len(X) * 0.1)
    X_train = X[0:pos1]
    Y_train = Y[0:pos1]
    X_test = X[pos1:pos2]
    Y_test = Y[pos1:pos2]

    #cl.decision_trees(X_train, Y_train, X_test, Y_test, result_file, model)
    #cl.nb_classifier(X_train, Y_train, X_test, Y_test, result_file, model)
    cl.svm_ml(X_train, Y_train, X_test, Y_test, result_file, model)



if __name__ == '__main__':
    main()
