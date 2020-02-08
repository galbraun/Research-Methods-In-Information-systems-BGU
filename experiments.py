import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

sns.set()
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances


def plot_gridsearch_heatmap(grid, title='GridSearch'):
    sns.set_style("dark")

    axis_0_param, axis_1_param = grid.param_grid.keys()
    axis_0_values, axis_1_values = grid.param_grid[axis_0_param], grid.param_grid[axis_1_param]

    scores_std = np.array([f'{x} \n± {grid.cv_results_["std_test_score"][i]}' if len(str(x)) > 10 else f'{x} ± {grid.cv_results_["std_test_score"][i]}' for i, x in
                           enumerate(grid.cv_results_['mean_test_score'])])
    scores_labels = scores_std.reshape(len(axis_0_values), len(axis_1_values))
    scores_values = grid.cv_results_['mean_test_score'].reshape(len(axis_0_values), len(axis_1_values))

    df_annot = pd.DataFrame(scores_labels, index=axis_0_values, columns=axis_1_values)
    df_values = pd.DataFrame(scores_values, index=axis_0_values, columns=axis_1_values)

    f, ax = plt.subplots(figsize=(20, 20))
    print(df_values.values)
    sns.heatmap(df_values, annot=df_annot, linewidths=.5, ax=ax, fmt='', annot_kws={"size": 15, "color": 'red'}, cmap="YlGnBu")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    # ax.set_xlabel(axis_0_param)
    # ax.set_ylabel(axis_1_param)
    ax.set_xlabel(axis_1_param, fontsize=15)
    ax.set_ylabel(axis_0_param, fontsize=15)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_title(title, fontsize=20)

def plot_time_gridsearch_heatmap(grid, title='GridSearch'):
    sns.set_style("dark")

    axis_0_param, axis_1_param = grid.param_grid.keys()
    axis_0_values, axis_1_values = grid.param_grid[axis_0_param], grid.param_grid[axis_1_param]

    scores_std = np.array([f'{x} \n± {grid.cv_results_["std_fit_time"][i]}' if len(str(x)) > 10 else f'{x} ± {grid.cv_results_["std_fit_time"][i]}' for i, x in
                           enumerate(grid.cv_results_['mean_fit_time'])])
    scores_labels = scores_std.reshape(len(axis_0_values), len(axis_1_values))
    scores_values = grid.cv_results_['mean_fit_time'].reshape(len(axis_0_values), len(axis_1_values))

    df_annot = pd.DataFrame(scores_labels, index=axis_0_values, columns=axis_1_values)
    df_values = pd.DataFrame(scores_values, index=axis_0_values, columns=axis_1_values)

    f, ax = plt.subplots(figsize=(20, 20))
    print(df_values.values)
    sns.heatmap(df_values, annot=df_annot, linewidths=.5, ax=ax, fmt='', annot_kws={"size": 15, "color": 'red'}, cmap="YlGnBu")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    # ax.set_xlabel(axis_0_param)
    # ax.set_ylabel(axis_1_param)
    ax.set_xlabel(axis_1_param, fontsize=15)
    ax.set_ylabel(axis_0_param, fontsize=15)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_title(title, fontsize=20)

def isolation_forest_gridsearch(X, y):
    clf = GridSearchCV(estimator=IsolationForest(n_jobs=-1), scoring='accuracy', cv=5, refit=True,
                       return_train_score=True,
                       verbose=2,
                       param_grid={'max_samples': [500, 1000, 2000],
                                   'max_features': [0.2, 0.5, 1.0]})

    clf.fit(X, y)
    return clf


def knn_gridsearch(X, y):
    clf = GridSearchCV(estimator=KNeighborsClassifier(n_jobs=-1), scoring='average_precision', cv=5, refit=True,
                       return_train_score=True,
                       verbose=2,
                       param_grid={'n_neighbors': [3, 4, 5, 6],
                                   'p': [1, 2, 3]})

    clf.fit(X, y)
    return clf


def random_forest_gridsearch(X, y):
    clf = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1), scoring='average_precision', cv=5, refit=True,
                       return_train_score=True,
                       verbose=2,
                       param_grid={'max_depth': [3, 5, 8],
                                   'n_estimators': [50, 100, 250, 500]})

    clf.fit(X, y)
    return clf

def logistic_regression_gridsearch(X, y):
    clf = GridSearchCV(estimator=LogisticRegression(n_jobs=-1), scoring='average_precision', cv=5, refit=True,
                       return_train_score=True,
                       verbose=2,
                       param_grid={'C': [1, 10, 100, 1000, 10000],
                                   'penalty': ['l1', 'l2']})

    clf.fit(X, y)
    return clf


def get_hybrid_features(X):
    kmeans = KMeans(n_clusters=6, n_jobs=-1).fit(X)

    max_data = []
    min_data = []
    mean_data = []

    for i in set(kmeans.labels_):
        cluster_data = np.asarray(X.values[np.where(kmeans.labels_ == i)[0]])
        max_dist = pairwise_distances(X, cluster_data).max(axis=1)
        max_data.append(max_dist)
        min_dist = pairwise_distances(X, cluster_data).min(axis=1)
        min_data.append(min_dist)
        mean_dist = pairwise_distances(X, cluster_data).mean(axis=1)
        mean_data.append(mean_dist)

    max_data = np.array(max_data).T
    min_data = np.array(min_data).T
    mean_data = np.array(mean_data).T

    distance_from_center = pairwise_distances(X, kmeans.cluster_centers_)

    return np.hstack([max_data, min_data, mean_data, distance_from_center])

def prepare_hybrid_data(X):
    return np.hstack([X, get_hybrid_features(X)])
