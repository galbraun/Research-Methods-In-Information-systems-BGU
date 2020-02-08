from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN

RANDOM_STATE = 42

def train_isolation_forest(X, params):
    clf = IsolationForest(verbose=2, n_jobs=-1).fit(X)
    pass

def train_OCSVM(X, params):
    clf = OneClassSVM(**params).fit(X)
    pass

def train_hdbscan(X, params):
    clf = HDBSCAN(X, params)
    pass

def train_KMeans(X, params):
    kmeans = KMeans(**params).fit(X)
    pass

def train_hierarchical_clustering():

    pass

def check_hyperparameters():
    pass