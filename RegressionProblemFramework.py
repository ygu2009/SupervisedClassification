"""
For exmaple:
Data:
    - train.txt: 10,000 records x 255 features + 1 target 
    - test.txt : 1,000 records x 255 features            
The feature variables could contrain numberical values, categorical values(eg, strings), and missing values.

Library: scikit-learn (Machine Learning in Python http://scikit-learn.org/stable/)

Implemented in Python 2.7 by Yingying Gu.
Date: 04/02/2016
"""

import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import cross_validation
import matplotlib.pyplot as plt


def load_data(filename):
    """
    The function is to read the given data file
    :param path:
    :return:
    """
    contents = []
    with open(filename,'r') as f:
        first_line = f.readline()[:-1].split('\t')  # skip the first description line
        for line in f:
            t=line[:-1].split('\t')  # split each number between '\t' and remove '\n' in the end of each line
            tmp = []
            for item in t:
                try:
                    item = float(item)
                    tmp.append(item)   # string number to float value
                except:
                    if not item:    # missing value
                        tmp.append(np.NAN)  # change missing data ''  to NAN(not a number)
                    else:
                        tmp.append(item)  # keep the string and empty value
            contents.append(tmp)
    return first_line, contents

def preprocessing_transform(data):
    """
    Transform the categorical number to int (Encode the string as int)
    :return: numerical data
    """
    # categorical number to int, e.g. 100th feature
    data[:, 100] = preprocessing.LabelEncoder().fit_transform(data[:,100])
    return data

def model_selection(estimator_name):
    """
    ---------------------------
    - Select Regression Model -
    ---------------------------
    :return:

    """
    if estimator_name == "ada":
        return AdaBoostRegressor(DecisionTreeRegressor(max_depth=20), n_estimators=300, learning_rate=1.0,
                                 loss='square', random_state=None)

    elif estimator_name == 'rf':
        return RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=None, min_samples_split=2,
                          min_samples_leaf=10, min_weight_fraction_leaf=0.0, max_features='auto',
                          max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1,
                          random_state=None, verbose=0, warm_start=False)

    elif estimator_name == 'svr_rbf':
        return SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.01, C=100.0, epsilon=0.1, shrinking=True,
              cache_size=200, verbose=False, max_iter=-1)

    elif estimator_name == 'KNN':
        return KNeighborsRegressor(n_neighbors=10, weights='distance', algorithm='auto', leaf_size=30, p=2,
                                   metric='minkowski', metric_params=None, n_jobs=-1)

    elif estimator_name == 'LR':
        return LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
        # return Lasso(alpha=1.0, fit_intercept=True, normalize=True, precompute=False, copy_X=True, max_iter=1000,
        #              tol=0.01, warm_start=False, positive=False, random_state=None, selection='cyclic')

    elif estimator_name == "Bayesian":
        return BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06,
                             compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)

    else:
        print "please select your estimator: ada, rf, svr_rbf or LR."
        return 0

def feature_selection(feature_weight):
    """
    The function is for evaluating feature importance from the regression model results
    :return:
    """
    # plot the feature importance
    weights = 0
    N = len(feature_weight)
    for i in range(N):
        weights = weights+feature_weight[i]
    weights = np.array(weights) / N  # normalization

    # output the top ranked importance
    feature_names = np.array(names[1:], dtype=object)
    sorted_top_features_name = feature_names[(-weights).argsort()]
    sorted_top_features_weight = weights[(-weights).argsort()]
    for i in range(15):
        print "rank %d: name: %s, importance: %0.3f" % (i+1, sorted_top_features_name[i], sorted_top_features_weight[i])

    # plot the feature importance
    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(train_set_x.shape[1]), weights, color="r", align="center")
    plt.xlim([-1, train_set_x.shape[1]])
    plt.show()

    index = list(np.where(weights>0.005))
    print index

def evaluation_MSE(y_test, y_pred):
    """
    Evaluate the prediction with ground truth by using mean square error
    :return:
    """
    MSE = mean_squared_error(y_test, y_pred)
    return MSE

def evaluation_R2(y_test, y_pred):
    """
    Evaluate the prediction with ground truth by using R squared error
    :return:
    """
    R2 = r2_score(y_test, y_pred)
    return R2

def evaluation_by_cross_validataion(cv, X_train_set, y_train_set):
    avg_MSE = 0
    avg_R2 = 0
    nn = 0
    feature_weight = [] # for keep the feature weights from each fold evaluation
    for train_index, test_index in cv:
        nn = nn + 1  # the nn-fold in cv

        X_train_cv, X_test_cv = X_train_set[train_index], X_train_set[test_index]
        y_train_cv, y_test_cv = y_train_set[train_index], y_train_set[test_index]

        # data normalization
        scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train_cv)
        X_train_cv = scaler.transform(X_train_cv)
        X_test_cv = scaler.transform(X_test_cv)

        # choose the regression estimator: ada, rf, svr_rbf, KNN or LR
        estimator_name = 'svr_rbf'
        clf_cv = model_selection(estimator_name)

        # fit regression model
        model_cv = clf_cv.fit(X_train_cv, y_train_cv)

        # prediction based on trained model
        y_pred_cv = model_cv.predict(X_test_cv)

        # feature importance analysis
        # feature_weight.append(clf.feature_importances_)  # features weights from classifier

        # evaluation/scoring by using MSE and R2
        MSE = evaluation_MSE(y_test_cv, y_pred_cv)
        R2 = evaluation_R2(y_test_cv, y_pred_cv)

        print '%d-fold: MSE: %f ; R squared: %f' % (nn, MSE, R2)

        avg_MSE += MSE
        avg_R2 += R2

    # feature importance analysis after using "clf.feature_importances" which can be accessed in your regression model
    # feature_selection(feature_weight)

    print '=====', estimator_name, '====='
    print 'Average MSE: %f' % (avg_MSE/nn)
    print 'Average R squared: %f' % (avg_R2/nn)
    print '=============================='

if __name__ == '__main__':

    # load and clean the training data
    filename_train = './train.txt'
    names, data = load_data(filename_train)  # load/read the data from files
    data = np.array(data, dtype=object) # save the list into numpy array
    train_data, train_set_y = data[:,1:], data[:,0]  # separate the target from all data, and save it into train_set_y
    train_set_x = preprocessing_transform(train_data.copy())  # transform the categorical value and missing value

    # Load and clean the testing data
    filename_test = './test.txt'
    test_names, test_data = load_data(filename_test)
    test_data = np.array(test_data, dtype=object)
    test_set_x = preprocessing_transform(test_data.copy())

    # Imputation of missing values
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0).fit(train_set_x)
    train_set_x = imp.transform(train_set_x)
    test_set_x = imp.transform(test_set_x)

    # top 15 features from feature importance analysis
    top_index = np.array([...])

    X_train = train_set_x[:, top_index[0:5]]
    y_train = train_set_y

    X_test = test_set_x[:, top_index[0:5]]

    # cross validation to split the training and testing data set by using n_folds, e.g. n_folds=10
    N = 10
    cv = cross_validation.KFold(n=X_train.shape[0], n_folds=N, shuffle=True, random_state=None)
    # for selecting important features, tuning the parameters
    evaluation_by_cross_validataion(cv, X_train, y_train)

    # data normalization
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # choose the regression estimator: ada, rf, svr_rbf, KNN or LR
    clf = model_selection('svr_rbf')

    # fit regression model
    model = clf.fit(X_train, y_train)

    # prediction based on trained model
    y_pred = model.predict(X_test)

    # Write to text file which contains 1 prediction per line for each record in the test dataset
    output = open('test_predx.txt', 'w')
    for i in range(y_pred.shape[0]):
        output.write(str(y_pred[i]))
        output.write("\n")
    output.close()





