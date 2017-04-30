
To run ipython notebook please install below libraries:

import sys
import os
import csv
import numpy as np
import scipy.io.wavfile
from sklearn.utils import shuffle
import cvxopt
import  scipy as sp
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


I have also used the Ulitilty module from the qeexo examples:
from utility import load_instances, load_labels, load_timestamps, convert_to_classlabels, write_results

And also import the my own SVM module :

from SVM import SVM

