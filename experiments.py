import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

sns.set()
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances


def plot_gridsearch_heatmap(grid, title='GridSearch'):
    sns.set_style("dark")

    axis_0_param, axis_1_param = grid.param_grid.keys()
    axis_0_values, axis_1_values = grid.param_grid[axis_0_param], grid.param_grid[axis_1_param]
    scores = grid.cv_results_['mean_test_score'].reshape(len(axis_0_values),
                                                         len(axis_1_values))
    df = pd.DataFrame(scores, index=axis_0_values, columns=axis_1_values)
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df, annot=True, linewidths=.5, ax=ax)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xlabel(axis_0_param)
    ax.set_ylabel(axis_1_param)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_title(title)


def isolation_forest_gridsearch(data):
    clf = GridSearchCV(estimator=IsolationForest(n_jobs=-1), scoring='average_precision', cv=5, refit=True,
                       return_train_score=True,
                       verbose=2,
                       param_grid={'max_samples': [500, 1000, 2000],
                                   'max_features': [0.2, 0.5, 1.0]})

    clf.fit(data['X_train'], data['y_train'])
    return clf


def random_forest_gridsearch(data):
    clf = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1), scoring='average_precision', cv=5, refit=True,
                       return_train_score=True,
                       verbose=2,
                       param_grid={'max_depth': [1, 2, 3],
                                   'n_estimators': [1, 2, 3]})

    clf.fit(data['X_train'], data['y_train'])
    return clf


def get_hybrid_features(data):
    kmeans = KMeans(n_clusters=6, n_jobs=-1).fit(data['X_train'])
    return pairwise_distances(data['X_train'].values, kmeans.cluster_centers_)


def prepare_hybrid_data(data):
    return np.hstack([data['X_train'], get_hybrid_features(data)])
