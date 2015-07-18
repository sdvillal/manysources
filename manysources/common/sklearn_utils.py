"""Integrating sklearn in our clumsy framework, the clumsy way.
How to add a new model:
  - Identify SKLearn class
  - Add it to the Scikit2Short dictionary
  - Create a shortName_default_params() and a shortName_nonid_params() method
To instantiate:
  sk_factory(skclass) or sk_factory(shortName)
"""

#TODO GLMnet
#TODO: SS, NaiveBayes, LDA/QDA

from collections import OrderedDict
from sklearn.base import BaseEstimator
from oscail.common.config import Configurable, Configuration
from rdkit.ML.NaiveBayes.ClassificationModel import NaiveBayesClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor, \
    ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble.forest import RandomForestClassifier
import numpy as np
from sklearn.gaussian_process.gaussian_process import MACHINE_EPSILON
from sklearn.linear_model.ridge import Ridge
from sklearn.linear_model import Lasso, ElasticNet, Lars, OrthogonalMatchingPursuit, BayesianRidge, \
    ARDRegression, LogisticRegression, SGDClassifier, SGDRegressor, Perceptron, LassoLars, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, \
    KNeighborsRegressor, RadiusNeighborsRegressor, NearestCentroid
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA, PLSSVD
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

Scikit2Short = {
    #Ensembles
    RandomForestClassifier: 'rfc',
    RandomForestRegressor: 'rfr',
    GradientBoostingClassifier: 'gbc',
    GradientBoostingRegressor: 'gbr',
    ExtraTreesClassifier: 'etc',
    ExtraTreesRegressor: 'etr',
    #NaiveBayes
    NaiveBayesClassifier: 'nb',  # TODO: parameters
    GaussianNB: 'gnb',
    #GLMs
    Ridge: 'ridge',
    Lasso: 'lasso',
    ElasticNet: 'elnet',
    Lars: 'lars',
    OrthogonalMatchingPursuit: 'omp',
    BayesianRidge: 'bayridge',
    ARDRegression: 'ardr',
    LogisticRegression: 'logreg',
    SGDClassifier: 'sgdc',
    SGDRegressor: 'sgdr',
    Perceptron: 'perceptron',
    LassoLars: 'lassolars',
    LinearRegression: 'lr',
    #Support Vector Machines
    SVC: 'svc',
    NuSVC: 'nusvc',
    LinearSVC: 'linsvc',
    SVR: 'svr',
    NuSVR: 'nusvr',
    #NearestNeighbours
    KNeighborsClassifier: 'knc',
    RadiusNeighborsClassifier: 'rnc',
    KNeighborsRegressor: 'knr',
    RadiusNeighborsRegressor: 'rnr',
    NearestCentroid: 'nc',
    #Gaussian Processes
    GaussianProcess: 'gp',
    #Partial Least Squares
    PLSRegression: 'plsr',
    PLSCanonical: 'plscan',
    CCA: 'cca',
    PLSSVD: 'plssvd',
    #Decision Trees
    DecisionTreeClassifier: 'dtc',
    DecisionTreeRegressor: 'dtr'
}
Short2Scikit = dict((v, k) for k, v in Scikit2Short.iteritems())


class ScikitAdaptor(BaseEstimator, Configurable):

    def __init__(self,
                 skclassifier=RandomForestClassifier(random_state=0),
                 configuration_dict=None,
                 non_id_params=(),
                 short_name=None,
                 **kwargs):
        super(ScikitAdaptor, self).__init__()
        self.classifier = skclassifier
        self.configuration_dict = configuration_dict if configuration_dict else skclassifier.get_params()
        self.non_id_params = non_id_params
        self.short_name = short_name if short_name else \
            Scikit2Short.get(self.classifier.__class__, self.classifier.__class__.__name__)

    def configuration(self):
        return Configuration(name=self.short_name,
                             configuration_dict=self.configuration_dict,
                             non_id_keys=self.non_id_params)

    def fit(self, X, y):
        return self.train(X, y)

    def train(self, X, y):
        classifier = self.classifier.fit(X, y)
        if not classifier:
            classifier = self.classifier
        self.classifier = classifier
        return self

    def score(self, X):
        return self.scores(X)[:, 1]

    def scores(self, X):
        try:
            scores = self.classifier.predict_proba(X)  # TODO: some have methods with further refinements,
                                                       #    like allowing to use a subset of classifiers in ensembles
        except:  # (AttributeError, TypeError)
            scores = self.classifier.predict(X)
        if scores.ndim < 2:
            scores2 = np.zeros((len(scores) if isinstance(scores, np.ndarray) else 1, 2))  # FIXME: isinstance is slow
            scores2[:, 1] = scores  # ATM, target class = 1
            return scores2
        return scores

    def predict_proba(self, X):
        return self.scores(X)


def get_skl_default_params(short_name):
    return eval('%s_default_params()' % short_name)


def sk_wrapper(skclass, configuration_dict=None, non_ids=None):
    if isinstance(skclass, str):
        skclass = Short2Scikit[skclass]
    short_name = Scikit2Short[skclass]
    if not configuration_dict:
        configuration_dict = eval('%s_default_params()' % short_name)
    if not non_ids:
        non_ids = eval('%s_nonid_params()' % short_name)
    return skclass(**configuration_dict), configuration_dict, non_ids, short_name


def skl_factory(skclass, configuration_dict=None):
    return ScikitAdaptor(*sk_wrapper(skclass, configuration_dict=configuration_dict))

##########################
#Configuration (ID stuff)
#We usually avoid introspection to get finer control on the parameters order and defaults
##########################


######RandomForests

def rfc_default_params():
    return OrderedDict((
        ('n_estimators', 10),
        ('criterion', 'gini'),
        ('max_depth', None),
        ('min_samples_split', 1),
        ('min_samples_leaf', 1),
        ('min_density', 0.1),
        ('max_features', 'auto'),
        ('bootstrap', True),
        ('compute_importances', False),
        ('oob_score', False),
        ('n_jobs', 1),
        ('random_state', 0),
        ('verbose', 0)))


def rfc_nonid_params():
    return 'verbose', 'n_jobs'  # OOB, importances change the internal representation of the classifier


def rfr_default_params():
    params = rfc_default_params()
    params['criterion'] = 'mse'
    return params


def rfr_nonid_params():
    return rfc_nonid_params()


######GradientBoosting

def gbc_default_params():
    return OrderedDict((
        ('learn_rate', 0.1),
        ('loss', 'deviance'),  # Or 'ls' or 'lad' for regression
        ('subsample', 1.0),
        ('n_estimators', 100),
        #DecisionTree params
        ('min_samples_split', 1),
        ('min_samples_leaf', 1),
        ('max_depth', 3),
        #Rng
        ('random_state', 0)))


def gbc_nonid_params():
    return ()


def gbr_default_params():
    params = gbc_default_params()
    params['loss'] = 'ls'
    return params


def gbr_nonid_params():
    return ()


######Extremely Randomized Trees Classifier

def etc_default_params():
    return OrderedDict((
        ('n_estimators', 10),
        ('criterion', 'gini'),     # or 'entropy'
        ('max_depth', None),
        ('min_samples_split', 1),
        ('min_samples_leaf', 1),
        ('min_density', 0.1),
        ('max_features', 'auto'),  # or 'sqrt' or 'log2' or 'none'
        ('bootstrap', False),      # whether bootstrap samples are used when building trees
        ('compute_importances', True),
        ('oob_score', False),      # whether to use out-of-bag samples to estimate the generalization error
        ('n_jobs', 1),
        ('random_state', 0),
        ('verbose', 0)
    ))


def etc_nonid_params():
    return 'n_jobs', 'verbose'  # 'compute_importances'


######Extremely Randomized Trees Regressor

def etr_default_params():
    return OrderedDict((
        ('n_estimators', 10),
        ('criterion', 'mse'),
        ('max_depth', None),
        ('min_samples_split', 1),
        ('min_samples_leaf', 1),
        ('min_density', 0.1),
        ('max_features', 'auto'),
        ('bootstrap', False),
        ('compute_importances', True),
        ('oob_score', False),
        ('n_jobs', 1),
        ('random_state', None),
        ('verbose', 0)
    ))


def etr_nonid_params():
    return 'compute_importances', 'n_jobs', 'verbose'


###############################
# Generalized Linear Models
###############################
# TODO: glmnet (e.g. in MDP)
###############################

######Ridge

def ridge_default_params():
    return OrderedDict((
        ('alpha', 1.0),
        ('fit_intercept', True),
        ('normalize', False),
        ('copy_X', True),
        ('tol', 1e-3)
    ))


def ridge_nonid_params():
    return 'copy_X'


######Lasso

def lasso_default_params():
    return OrderedDict((
        ('alpha', 1.0),
        ('fit_intercept', True),
        ('normalize', False),
        ('precompute', 'auto'),
        ('copy_X', True),
        ('max_iter', 1000),
        ('tol', 1e-4),
        ('warm_start', False),
        ('positive', False)
    ))


def lasso_nonid_params():
    return 'copy_X'


######ElasticNet

def elnet_default_params():
    return OrderedDict((
        ('alpha', 1.0),  # constant that multiplies the penalty term
        ('rho', 0.5),    # For rho = 0 the penalty is an L1 penalty. For rho = 1 it is an L2 penalty.
                         # For 0 < rho < 1, the penalty is a combination of L1 and L2
        ('fit_intercept', True),
        ('normalize', False),
        ('precompute', 'auto'),
        ('max_iter', 1000),
        ('copy_X', True),
        ('tol', 1e-4),
        ('warm_start', False),
        ('positive', False)
    ))


def elnet_nonid_params():
    return 'copy_X'


######LeastAngleRegression #TODO check Lasso_path

def lars_default_params():
    return OrderedDict((
        ('n_nonzero_coefs', 500),
        ('fit_intercept', True),
        ('verbose', False),
        ('normalize', True),
        ('precompute', 'auto'),
        ('copy_X', True),
        ('eps', np.finfo(np.float).eps)  # The machine-precision regularization in the computation of the
                                         # Cholesky diagonal factors.
    ))


def lars_nonid_params():
    return 'verbose', 'copy_X'


######OrthogonalMatchingPursuit

def omp_default_params():
    return OrderedDict((
        ('n_nonzero_coefs', None),
        ('tol', None),
        ('fit_intercept', True),
        ('normalize', True),
        ('precompute_gram', False),
        ('copy_X', True),
        ('copy_Gram', True),
        ('copy_Xy', True)
    ))


def omp_nonid_params():
    return 'copy_X', 'copy_Xy', 'copy_Gram'


######BayesianRidge regression

def bayridge_default_params():
    return OrderedDict((
        ('n_iter', 300),
        ('tol', 1.e-3),
        ('alpha_1', 1e-6),  # shape parameter for the Gamma distribution prior over the alpha parameter.
        ('alpha_2', 1e-6),  # inverse scale parameter (rate parameter) for the Gamma distribution prior
                            # over the alpha parameter
        ('lambda_1', 1e-6),  # shape parameter for the Gamma distribution prior over the lambda parameter
        ('lambda_2', 1e-6),  # inverse scale parameter (rate parameter) for the Gamma distribution prior
                             # over the lambda parameter
        ('compute_score', False),
        ('fit_intercept', True),
        ('normalize', False),
        ('copy_X', True),
        ('verbose', False)
    ))


def bayridge_nonid_params():
    return 'copy_X', 'verbose'


######ARDRegression

def ardr_default_params():
    return OrderedDict((
        ('n_iter', 300),
        ('tol', 1e-3),
        ('alpha_1', 1e-6),
        ('alpha_2', 1e-6),
        ('lambda_1', 1e-6),
        ('lambda_2', 1e-6),
        ('compute_score', False),
        ('threshold_lambda', 1e+4),
        ('fit_intercept', True),
        ('normalize', False),
        ('copy_X', True),
        ('verbose', False)
    ))


def ardr_nonid_params():
    return 'copy_X', 'verbose'


######LogisticRegression


def logreg_default_params():
    return OrderedDict((
        ('penalty', 'l2'),
        ('dual', False),
        ('tol', 1e-4),
        ('C', 1.0),  # Specifies the strength of the regularization. The smaller it is the bigger in the regularization.
                     # If None then C is set to n_samples.
        ('fit_intercept', True),
        ('intercept_scaling', 1),
        ('class_weight', None),
        ('random_state', 0),
    ))


def logreg_nonid_params():
    return ()


######SGDRegression

def sgdr_default_params():
    return OrderedDict((
        ('loss', 'squared_loss'),
        ('penalty', 'l2'),
        ('alpha', 0.0001),  # Constant that multiplies the regularization term
        ('rho', 0.85),      # The Elastic Net mixing parameter, with 0 < rho <= 1.
        ('fit_intercept', True),
        ('n_iter', 5),
        ('shuffle', False),  # Whether or not the training data should be shuffled after each epoch
        ('verbose', 0),
        ('p', 0.1),  # Epsilon in the epsilon-insensitive huber loss function
        ('seed', 0),  # seed to use when shuffling the data
        ('learning_rate', 'invscaling'),
        ('eta0', 0.01),  # initial learning rate
        ('power_t', 0.25),  # The exponent for inverse scaling learning rate
        ('warm_start', False),
    ))


def sgdr_nonid_params():
    return 'verbose'


######SGDClassification

def sgdc_default_params():
    return OrderedDict((
        ('loss', 'hinge'),
        ('penalty', 'l2'),
        ('alpha', 0.0001),
        ('rho', 0.85),
        ('fit_intercept', True),
        ('n_iter', 5),
        ('shuffle', False),
        ('verbose', 0),
        ('n_jobs', 1),
        ('seed', 0),
        ('learning_rate', 'optimal'),
        ('eta0', 0.0),
        ('power_t', 0.5),
        ('class_weight', None),
        ('warm_start', False)
    ))


def sgdc_nonid_params():
    return 'verbose', 'n_jobs'


######Perceptron

def perceptron_default_params():
    return OrderedDict((
        ('penalty', None),
        ('alpha', 0.0001),
        ('fit_intercept', True),
        ('n_iter', 5),
        ('shuffle', False),
        ('verbose', 0),
        ('eta0', 1.0),
        ('n_jobs', 1),
        ('seed', 0),
        ('class_weight', None),
        ('warm_start', False)
    ))


def perceptron_nonid_params():
    return 'verbose', 'n_jobs'


######LassoLars

def lassolars_default_params():
    return OrderedDict((
        ('alpha', 1.0),
        ('fit_intercept', True),
        ('verbose', False),
        ('normalize', True),
        ('precompute', 'auto'),
        ('max_iter', 500),
        ('eps', np.finfo(np.float).eps),
        ('copy_X', True)
    ))


def lassolars_nonid_params():
    return 'verbose', 'copy_X'


######LinearRegression

def lr_default_params():
    return OrderedDict((
        ('fit_intercept', True),
        ('normalize', False),
        ('copy_X', True)
    ))


def lr_nonid_params():
    return 'copy_X'


###############################
# Support Vector Marchines
###############################

######SVC

def svc_default_params():
    return OrderedDict((
        ('C', None),
        ('kernel', 'rbf'),   # It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
        ('degree', 3.0),     # Only significant if polykernel or sigmoid kernel are used
        ('gamma', 0.0),      # Kernel coefficient for 'rbf' and 'poly'.
                             # If gamma is 0.0 then 1/n_features will be used instead
        ('coef0', 0.0),      # Only significant if polykernel or sigmoid kernel are used
        ('probability', False),
        ('shrinking', True),
        ('tol', 1e-3),
        ('cache_size', 200),
        ('class_weight', None),  # Set the parameter C of class i to class_weight[i]*C. If not given, all classes are
                                 # supposed to have weight=1. The 'auto' mode uses the values of y to automatically
                                 # adjust weights inversely proportional to class frequencies.
        ('verbose', False)
    ))


def svc_nonid_params():
    return 'cache_size', 'verbose'


######NuSVC

def nusvc_default_params():
    return OrderedDict((
        ('nu', 0.5),  # in [0,1]. An upper bound on the fraction of training errors and a lower bound of the fraction
                      # of support vectors.
        ('kernel', 'rbf'),
        ('degree', 3),
        ('gamma', 0.0),
        ('coef0', 0.0),
        ('probability', False),
        ('shrinking', True),
        ('tol', 1e-3),
        ('cache_size', 200),
        ('class_weight', None),
        ('verbose', False)
    ))


def nusvc_nonid_params():
    return 'cache_size', 'verbose'


######LinearSVC

def linsvc_default_params():
    return OrderedDict((
        ('C', 1.0),   # Penalty parameter C of the error term. If None then C is set to n_samples (does not wort atm).
        ('loss', 'l2'),
        ('penalty', 'l2'),
        ('dual', True),
        ('tol', 1e-4),
        ('multi_class', 'ovr'),   # Determines the multi-class strategy if `y` contains more than two classes.
        ('fit_intercept', True),
        ('intercept_scaling', 1.0),
        ('class_weight', None),
        ('verbose', 0.0)
    ))


def linsvc_nonid_params():
    return 'verbose'


######Support Vector Regression

def svr_default_params():
    return OrderedDict((
        ('C', 1.0),   # None does not work
        ('epsilon', 0.1),   # It specifies the epsilon-tube within which no penalty is associated in the training loss
                            # function with points predicted within a distance epsilon from the actual value.
        ('kernel', 'rbf'),
        ('degree', 3),
        ('gamma', 0.0),
        ('coef0', 0.0),
        ('probability', False),
        ('shrinking', True),
        ('tol', 1e-3),
        ('cache_size', 200),
        ('verbose', False)
    ))


def svr_nonid_params():
    return 'cache_size', 'verbose'


######NuSVR

def nusvr_default_params():
    return OrderedDict((
        ('C', 1),   # None should work, but they have a bug
        ('nu', 0.5),
        ('kernel', 'rbf'),
        ('degree', 3),
        ('gamma', 0.0),
        ('coef0', 0.0),
        ('probability', False),
        ('shrinking', True),
        ('tol', 1e-3),
        ('cache_size', 200),
        ('verbose', False)
    ))


def nusvr_nonid_params():
    return 'cache_size', 'verbose'


###############################
# NearestNeigbours
###############################

######K Nearest Neighbours

def knc_default_parameters():
    return OrderedDict((
        ('n_neighbors', 5),
        ('weights', 'uniform'),
        ('algorithm', 'auto'),   # Can bee 'kd_tree', 'ball_tree', 'brute' or 'auto'
        ('leaf_size', 30),       # only for kd-tree and ball_tree algorithms
        ('warn_on_equidistant', True),
        ('p', 2)                 # Parameter for the Minkowski metric. P=2 -> euclidian distance
    ))


def knc_nonid_params():
    return 'warn_on_equidistant'


######Radius Neighbours

def rnc_default_params():
    return OrderedDict((
        ('radius', 1.0),
        ('weights', 'uniform'),
        ('algorithm', 'auto'),
        ('leaf_size', 30),
        ('p', 2),
        ('outlier_label', None)  # Label, which is given for outlier samples (samples with no neighbors on given radius)
                                 # If set to None, ValueError is raised, when outlier is detected.
    ))


def rnc_nonid_params():
    return 'outlier_label'   # WARNING: not sure about it --> check


######K Nearest Neighbours Regressor

def knr_default_params():
    return OrderedDict((
        ('n_neighbors', 5),
        ('weights', 'uniform'),
        ('algorithm', 'auto'),
        ('leaf_size', 30),
        ('warn_on_equidistant', True),
        ('p', 2)
    ))


def knr_nonid_params():
    return 'warn_on_equidistant'


######Radius Neighbours Regressor

def rnr_default_params():
    return OrderedDict((
        ('radius', 1.0),
        ('weights', 'uniform'),
        ('algorithm', 'auto'),
        ('leaf_size', 30),
        ('p', 2),
    ))


def rnr_nonid_params():
    return ()


######Nearest Centroid

def nc_default_params():
    return OrderedDict((
        ('metric', 'euclidean'),
        ('shrink_threshold', None)  # Threshold for shrinking centroids to remove feature
    ))


def nc_nonid_params():
    return ()

###############################
# GaussianProcesses
###############################


######Gaussian Process

def gp_default_params():
    return OrderedDict((
        ('regr', 'constant'),    # A regression function returning an array of outputs of the linear regression
                                 # functional basis. Among 'constant', 'linear', 'quadratic'.
        ('corr', 'squared_exponential'),  # A stationary autocorrelation function returning the autocorrelation between
                                          # two points x and x'. Can be 'absolute_exponential', 'squared_exponential',
                                          # 'generalized_exponential', 'cubic' or 'linear'
        ('beta0', None),
        ('storage_mode', 'full'),  # A string specifying whether the Cholesky decomposition of the correlation matrix
                                   # should be stored in the class
        ('verbose', False),
        ('theta0', 1e-1),          # The parameters in the autocorrelation model. If thetaL and thetaU are also
                                   # specified, theta0 is considered as the starting point for the maximum likelihood
                                   # estimation of the best set of parameters.
        ('thetaL', None),          # Lower bound on the autocorrelation parameters for maximum likelihood estimation.
        ('thetaU', None),          # Upper bound on the autocorrelation parameters for maximum likelihood estimation.
        ('normalize', True),
        ('nugget', 10. * MACHINE_EPSILON),  # Introduce a nugget effect to allow smooth predictions from noisy data.
        ('optimizer', 'fmin_cobyla'),
        ('random_start', 1),       # The number of times the Maximum Likelihood Estimation should be performed from a
                                   # random starting point.
        ('random_state', 0)     # The generator used to shuffle the sequence of coordinates of theta in the Welch
                                # optimizer. If an integer is given, it fixes the seed.
    ))


def gp_nonid_params():
    return 'storage_mode', 'verbose'


###############################
# Partial Least Squares
###############################

######PLSRegression

def plsr_default_params():
    return OrderedDict((
        ('n_components', 2),
        ('scale', True),
        ('max_iter', 500),
        ('tol', 1e-6),
        ('copy', True)
    ))


def plsr_nonid_params():
    return 'copy'


######PLSCanonical

def plscan_default_params():
    return OrderedDict((
        ('n_components', 2),
        ('scale', True),
        ('algorithm', 'nipals'),  # The algorithm used to estimate the weights. "nipals" or "svd"
        ('max_iter', 500),
        ('tol', 1e-6),
        ('copy', True)
    ))


def plscan_nonid_params():
    return 'copy'


######CCA

def cca_default_params():
    return OrderedDict((
        ('n_components', 2),
        ('scale', True),
        ('max_iter', 500),
        ('tol', 1e-6),
        ('copy', True)
    ))


def cca_nonid_params():
    return 'copy'


######PLSSVD

def plssvd_default_params():
    return OrderedDict((
        ('n_components', 2),
        ('scale', True)
    ))


def plssvd_nonid_params():
    return ()


###############################
# Decision Trees
###############################


######Decision Tree Classifier

def dtc_default_params():
    return OrderedDict((
        ('criterion', 'gini'),     # or 'entropy'
        ('max_depth', None),
        ('min_samples_split', 1),
        ('min_samples_leaf', 1),
        ('min_density', 0.1),      # This parameter controls a trade-off in an optimization heuristic.
                                   # It controls the minimum density of the `sample_mask`
        ('max_features', None),    # The number of features to consider when looking for the best split.
        ('compute_importances', False),
        ('random_state', 0)
    ))


def dtc_nonid_params():
    return ()  # 'compute_importances'


######Decision Tree Regressor
def dtr_default_params():
    return OrderedDict((
        ('criterion', 'mse'),
        ('max_depth', None),
        ('min_samples_split', 1),
        ('min_samples_leaf', 1),
        ('min_density', 0.1),
        ('max_features', None),
        ('compute_importances', False),
        ('random_state', 0)
    ))


def dtr_nonid_params():
    return ()  # 'compute_importances'


######Naive Bayes
def gnb_default_params():
    return OrderedDict()


def gnb_nonid_params():
    return ()  # 'compute_importances'

if __name__ == '__main__':
    print 'Done'