# Avoiding warning
import warnings

def warn(*args, **kwargs): pass
warnings.warn = warn
import numpy
# _______________________________
seed =456
numpy.random.seed(seed)

# Essential Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# _____________________________
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, \
    RandomForestClassifier,  \
    AdaBoostClassifier,    \
    GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
    roc_auc_score, \
    average_precision_score, \
    roc_curve, \
    f1_score, \
    recall_score, matthews_corrcoef, auc,cohen_kappa_score


'
### Remove columns there is all zero values.
v = []
for i in range(X.shape[1]):
    if not np.all(X[:, i] == 0):
        v.append(i)

X = X[:, v]


from sklearn.utils import shuffle
X, y = shuffle(X, y)  # Avoiding bias


# Step 06 : Scaling the feature
# ______________________________________________________________________________
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X = scale.fit_transform(X)

# ______________________________________________________________________________
# Step 04 : Encoding y :
# ______________________________________________________________________________
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
# ______________________________________________________________________________


# scikit-learn :
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier,  AdaBoostClassifier,    GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Names = ['DNN']
#Names = ['LXGB', 'RF', 'ERT', 'SVM']

Classifiers = [
    #SVC(probability=True),
    #lgb.LGBMClassifier(n_estimators=1),
    #lgb.LGBMClassifier(n_estimators=350),
    XGBClassifier(n_estimators=35),
    #RandomForestClassifier(n_estimators=20),
    #ExtraTreesClassifier(),
    #SVC(probability=True, C=16, kernel='rbf', gamma=0.1088),
    #lgb.LGBMClassifier(),
    #RandomForestClassifier(),
    #ExtraTreesClassifier(n_estimators=700),
    #ExtraTreesClassifier(n_estimators=200),
    #SVC(probability=True),
    # grnn(std = x, verbose = False),
    # GaussianNB(), #4
    # BaggingClassifier(), #5
    #MLPClassifier(hidden_layer_sizes=500, solver='sgd', alpha=0.01),
    #MLPClassifier(),
    #AdaBoostClassifier(), #7
     #GradientBoostingClassifier(n_estimators=300), #8
    # SVC(probability=True, C=16, kernel='rbf', gamma=0.1088),

  
    
]


def runClassifiers():
    i=0
    Results = []  # compare algorithms
    
        accuray = []
        auROC= []
        avePrecision = []
        F1_Score = []
        AUC = []
        MCC = []
        Recall = []
        mean_TPR = 0.0
        mean_FPR = np.linspace(0, 1, 100)
        CM = np.array([
            [0, 0],
            [0, 0],
        ], dtype=int)
        print(classifier.__class__.__name__)
        model = classifier
        for (train_index, test_index) in cv.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]
            y_train = y[train_index]
            y_test = y[test_index]
            model.fit(X_train, y_train)
            # Calculate ROC Curve and Area the Curve
            y_proba = model.predict_proba(X_test)[:, 1]
            FPR, TPR, _ = roc_curve(y_test, y_proba)
            mean_TPR += np.interp(mean_FPR, FPR, TPR)
            mean_TPR[0] = 0.0
            roc_auc = auc(FPR, TPR)
            y_artificial = model.predict(X_test)
            auROC.append(roc_auc_score(y_test, y_proba))
            accuray.append(accuracy_score(y_pred=y_artificial, y_true=y_test))
           print ("Accuracy" %f, accuracy) 
        