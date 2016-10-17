"""
Library: scikit-learn (Machine Learning in Python http://scikit-learn.org/stable/)
Library: myLibs

Implemented in Python 2.7 by Yingying Gu.
Date: 02/15/2015
"""
import h5py
import numpy as np
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from myLibs import MyBayesianGMM


def model_selection(estimator_name):
    """
    ---------------------------
    - Select Regression Model -
    ---------------------------

    """
    if estimator_name == "ada":
        return AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=500, learning_rate=1, algorithm="SAMME", random_state=10) # learning_rate=1 is best

    elif estimator_name == "GBM":
        return GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=300, subsample=1.0,
                                          min_samples_split=10, min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                                          max_depth=10, init=None, random_state=0, max_features='sqrt', verbose=0,
                                          max_leaf_nodes=None, warm_start=False, presort='auto')

    elif estimator_name == 'rf':
        return RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=None, min_samples_split=2,
                                      min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_features='sqrt',
                                      max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=10,
                                      verbose=0, warm_start=False, class_weight=None)

    elif estimator_name == 'svc_rbf':
        return SVC(C=1000.0, kernel='rbf', degree=1, gamma='auto', coef0=0.0, shrinking=False, probability=True,
                   tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                   decision_function_shape=None, random_state=None)

    elif estimator_name == 'KNN':
        return KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto', leaf_size=30, p=2,
                                    metric='minkowski', metric_params=None, n_jobs=1)

    elif estimator_name == 'LR':
        return linear_model.LogisticRegression(penalty='l1', dual=False, tol=0.1, C=10.0, fit_intercept=True,
                                               intercept_scaling=1, class_weight=None, random_state=10, solver='liblinear',
                                               max_iter=2000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

    elif estimator_name == "MyBayesian":
        return MyBayesianGMM.BayesianGMMClassifier()

    else:
        print "please select your estimator: ada, GBM, rf, svc_rbf, KNN, LR or MyBayesian."
        return 0


def evaluation_by_cross_validataion(cv, X_train_set, y_train_set):

    # Compute ROC curve and ROC area for each class
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'navy', 'gray', 'tomato']

    plt.close()
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0,1.1,0.1))
    ax.set_yticks(np.arange(0,1.1,0.1))
    plt.grid()

    nn = -1
    for train_index, test_index in cv:
        nn = nn + 1  # the nn-fold in cv

        X_train_cv, X_test_cv = X_train_set[train_index], X_train_set[test_index]
        y_train_cv, y_test_cv = y_train_set[train_index], y_train_set[test_index]

        # choose the classification model/estimator: ada, GBM, rf, svc_rbf, KNN, LR or MyBayesian
        estimator_name = 'svc_rbf'
        clf_cv = model_selection(estimator_name)

        # fit classification model on training data and make prediction based on trained model
        y_pred_cv = clf_cv.fit(X_train_cv, y_train_cv).predict_proba(X_test_cv)

        # plot the ROC curve for each fold in cv
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2): # two classes
            fpr[i], tpr[i], _ = roc_curve(y_test_cv, y_pred_cv[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[1], tpr[1], label='%d fold (AUC = %0.2f)' % (nn, roc_auc[1]), color = colors[nn])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")

    plt.show()

if __name__ == '__main__':

    # step 1: load the feature dataset (in "mat" file) which is already preprocessed from raw data.
    mat=h5py.File('./train_data.mat','r')
    train=np.array(mat.get('data'))

    mat2=h5py.File('./test_data.mat','r')
    test=np.array(mat2.get('data'))

    #'target', 'hist_mean', 'hist_var', 'obj_mode','obj_mass', 'obj_density', 'obj_nvoxels', 'ave_norm_of_gradient', 'hist_matrix(100bins)'
    # generate feature data matrix
    X_train=train[:,1:]
    y_train=train[:,0]

    X_test=test[:,:]

    # data normalization
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train_cv = scaler.transform(X_train)
    X_test_cv = scaler.transform(X_test)

    # cross validation:
    # split the training and testing data set by using n_folds, e.g. n_folds=10
    N = 10
    cv = cross_validation.KFold(n=X_train.shape[0], n_folds=N, shuffle=True, random_state=None)
    # for selecting important features, tuning the parameters
    evaluation_by_cross_validataion(cv, X_train, y_train)

    # choose the best classification model/estimator from cross validation
    estimator_name = 'svc_rbf'
    clf = model_selection(estimator_name)

    # train the classifier/model from all training data and make prediction on test data based on trained model
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)