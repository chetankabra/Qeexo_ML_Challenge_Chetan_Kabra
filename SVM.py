from __future__ import division
import numpy as np
import os
import numpy as np
import scipy.misc
import cvxopt
from sklearn.utils import shuffle

class SVM:

    def __init__(self):
        self.all_w = [];
        self.all_b =[];
        self.C = float(100)

    def fit(self, training_data, training_labels):
        n_samples, n_features= training_data.shape
        for each in set(training_labels):
            self.modify_training_labels(training_labels, each)
            X = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    X[i, j] = np.dot(training_data[i],training_data[j])

            P = cvxopt.matrix(np.outer(self.new_training_lables, self.new_training_lables.T) * X)
            q = cvxopt.matrix(np.ones(n_samples) * -1)
            A = cvxopt.matrix(self.new_training_lables, (1, n_samples))
            b = cvxopt.matrix(0.0)
            G_std = np.diag(np.ones(n_samples) * -1)
            G_slack = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_std, G_slack)))
            h_std = np.zeros(n_samples)
            h_slack = np.ones(n_samples) * self.C# soft margin calculation
            h = cvxopt.matrix(np.hstack((h_std, h_slack)))
            solution = cvxopt.solvers.qp(P, q, G, h, A, b)
            temp_alpha = np.array([])
            for e in solution['x'] :
                temp_alpha = np.hstack([temp_alpha,e])
            temp_alpha.reshape(len(solution['x']),1)
            support_vector_index = temp_alpha > 0.00
            self.alpha = temp_alpha[support_vector_index]
            self.support_vector_labels =  self.new_training_lables[support_vector_index]
            self.support_vector = training_data[support_vector_index]

            # Weight vector
            self.w = np.zeros(n_features)
            for n in range(len(self.alpha)):
                self.w += self.alpha[n] * self.support_vector_labels[n] * self.support_vector[n]
            self.all_w.append(self.w)
            #print (self.all_w.__len__())

            # calculating B value
            new_w = self.w.reshape(1,len(self.w))
            self.b = 0.0
            for index,yi in  enumerate(self.support_vector_labels):
                if yi > -1:
                    self.b = (1 - (np.dot(new_w, self.support_vector[index])))
                    break
            self.all_b.append(self.b)




    def modify_training_labels(self, training_labels, counter):
        self.old_training_labels = np.array(training_labels)
        self.old_training_labels[self.old_training_labels != counter] = -1
        self.old_training_labels[self.old_training_labels == counter] = 1
        self.new_training_lables = self.old_training_labels.astype('double')

    def predict(self, testing_data):
        #n_samples, n_features = testing_data.shape
        output = []
        for i,each in enumerate(testing_data):
            calculate = []
            for w in xrange(0,len(self.all_w)):
                current_prediction = np.dot(each,self.all_w[w]) + self.all_b[w]
                calculate.append(current_prediction)
            max_index = calculate.index(max(calculate))
            output.append(max_index+1)
        return output

    def accuracy(self, predict_labels, testing_labels):
        self.right_predication =0
        for i in range(len(predict_labels)):
            if (predict_labels[i] == testing_labels[i]):
                self.right_predication = self.right_predication + 1
        return (self.right_predication / len(predict_labels) *100)

    def create_training_labels(self,training_labels, counter):
        self.old_training_labels = training_labels
        end  = counter * 5
        start = end-5
        self.new_training_lables = np.ones(self.old_training_labels.__len__()) * -1
        self.new_training_lables[start:end]  = self.new_training_lables[start:end] * -1



def main(argv):
    raise Exception("This script isn't meant to be run.")

if __name__ == '__main__':
    exit(main(sys.argv[1:]))