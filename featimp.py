import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def spearman_sort(X, y, k=None):
    """Returns the sorted absolute ranks of each feature from X to y based on Spearman correlation"""
    X_copy = X.copy()
    X_copy['target'] = y
    ranks = np.abs(X_copy.corr(method='spearman')['target'])[:-1].sort_values(ascending=False) 
    ranks = [(ranks.index[i], val) if not np.isnan(val) else (ranks.index[i], 0) for i, val in enumerate(ranks)]
    if k is not None:
        return ranks[:k]
    else:
        return ranks


def PCA_sort(X, k=None):
    """Returns the sorted components (eigenvectors) of the matrix X. If k > 0, returns the top k components
       of the matrix X. Otherwise, returns the ranks of each feature of X from best to worst."""
    n = X.shape[1]
    
    scaled = StandardScaler().fit_transform(X.values)
    X_scaled = pd.DataFrame(scaled, index=X.index, columns=X.columns)
    
    X_matrix = X_scaled.to_numpy()
    X_mean = X_matrix - np.mean(X_matrix , axis = 0)
    cov_matrix = np.cov(X_mean, rowvar = False)
    eigenvalues , eigenvectors = np.linalg.eig(cov_matrix)

    explained_variance = eigenvalues / sum(eigenvalues)
    explained_var_ratio = np.cumsum(explained_variance)
    projection = np.abs(np.dot(eigenvectors.transpose(), np.eye(n).transpose()).transpose())
    contribution = np.sum(projection, axis=0)
    rank_contribution = np.argsort(contribution)[::-1]
    ranks = list(zip(list(X.columns[rank_contribution]), contribution[rank_contribution]))
    
    if k is not None:
        return ranks[:k]
    else:
        return explained_variance, explained_var_ratio, ranks
    
    
def importance_viz(ranks, size, adj):
    names = [tup[0] for tup in ranks][::-1]
    vals = [round(val[1], 4) for val in ranks][::-1]
    
    fig, ax = plt.subplots(figsize=size)
    ax.plot(vals, names, 'o')
    ax.hlines(names, [0], vals, linestyles='dotted', lw=2)

    for y, x in enumerate(vals):
        ax.annotate(str(x), xy=(x+adj, y), va='center')

    #Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    
    
def spearman_variance(X, y, size, n=100):
    data = X.copy()
    data['y'] = y
    rank_dict = {name: [] for name in X.columns}
    for _ in range(n):
        data_ = data.sample(len(data), replace=True)
        X_ = data_.drop(columns='y')
        y_ = data_['y']
        ranks = spearman_sort(X_, y_)
        for name, rank in ranks:
            rank_dict[name].append(rank)
    rank_mean_std = sorted([(name, np.mean(vals), np.std(vals)) for name, vals in rank_dict.items()], key=lambda x: x[1])
    features = [a[0] for a in rank_mean_std]
    means = [a[1] for a in rank_mean_std]
    err = [a[2] for a in rank_mean_std]
    
    fig, ax = plt.subplots(figsize=size)
    ax.set_title('Mean and Standard Deviation of feature importance')
    ax.barh(features, means, xerr = err)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
def pca_viz(explained_variance, explained_var_ratio, ranks, size):
    
    names = [val[0] for val in ranks][::-1]
    vals = [round(val[1],4) for val in ranks][::-1]
    fig, ax = plt.subplots(1, 2, figsize=size)

    ax[0].set_title('PCA Analysis')
    ax[0].set_xlabel('Principal Component')
    ax[0].set_ylabel('% Variance Explained')
    ax[0].set_xlim(0, len(explained_var_ratio)+1)
    ax[0].set_ylim(0, 105)
    ax[0].plot(range(1, len(explained_var_ratio)+1) ,explained_var_ratio*100, 'o-', c='black')
    ax[0].bar(range(1, len(explained_variance)+1), explained_variance*100)

    ax[1].set_title('Contribution of each feature')
    #ax[1].set_xlabel('Overall Contribution')
    #ax[1].set_ylabel('Feature Name')
    ax[1].plot(vals, names, 'o')
    ax[1].hlines(names, [0], vals, linestyles='dotted', lw=2)
    for y, x in enumerate(vals):
        ax[1].annotate(str(x), xy=(x+0.1, y), va='center')

    #Remove borders
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].xaxis.set_visible(False)

    plt.tight_layout()
    
    
def autosearch(model, X_train, y_train, X_val, y_val, method):
    
    N = len(y_val)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    baseline = sum(y_val == y_pred) / N  # Accuracy score
    features = list(X_train.columns)
    X_train_ = X_train.copy()
    X_val_ = X_val.copy()
    drop_features = []
    
    if method == 'spearman':
        for _ in range(len(features)-1):
            ranks = spearman_sort(X_train_, y_train)
            lowest_feat = ranks[-1][0]
            X_train_ = X_train_.drop(columns=lowest_feat)
            X_val_ = X_val_.drop(columns=lowest_feat)
            model_ = clone(model)
            model_.fit(X_train_, y_train)
            new_pred = model_.predict(X_val_)
            m = sum(y_val == new_pred) / N
            if baseline > m:
                return list(set(features) - set(drop_features)), baseline
            baseline = m
            drop_features.append(lowest_feat)
        return list(set(features) - set(drop_features)), baseline
    
    if method == 'pca':
        for _ in range(len(features)-1):
            ev, evr, ranks = PCA_sort(X_train_)
            lowest_feat = ranks[-1][0]
            X_train_ = X_train_.drop(columns=lowest_feat)
            X_val_ = X_val_.drop(columns=lowest_feat)
            model_ = clone(model)
            model_.fit(X_train_, y_train)
            new_pred = model_.predict(X_val_)
            m = sum(y_val == new_pred) / N
            if baseline > m:
                return list(set(features) - set(drop_features)), baseline
            baseline = m
            drop_features.append(lowest_feat)
        return list(set(features) - set(drop_features)), baseline
    

def dropcol_importance(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    baseline = sum(y_val == y_pred) / len(y_val)
    imp = []
    for col in X_train.columns:
        X_train_ = X_train.drop(col, axis=1)
        X_val_ = X_val.drop(col, axis=1)
        model_ = clone(model)
        model_.fit(X_train_, y_train)
        y_pred_ = model_.predict(X_val_)
        m = sum(y_val == y_pred_) / len(y_val)
        imp.append((col, baseline - m))
    return sorted([(a[0], np.abs(a[1])) for a in imp], key=lambda x: x[1], reverse=True)


def permutation_importance(model, X_val, y_val):
    y_pred = model.predict(X_val)
    baseline = sum(y_val == y_pred) / len(y_val)
    imp = []
    for col in X_val.columns:
        save_col = X_val[col].copy()
        X_val[col] = np.random.permutation(X_val[col])
        y_pred_ = model.predict(X_val)
        m = sum(y_val == y_pred_) / len(y_val)
        X_val[col] = save_col
        imp.append((col, baseline - m))
    return sorted([(a[0], np.abs(a[1])) for a in imp], key=lambda x: x[1], reverse=True)


def compare_importance(model, X, y, metric, k=8, folds=5):
    all_acc = []
    for func in [spearman_sort, PCA_sort, dropcol_importance, permutation_importance]:
        if func is spearman_sort:
            ranks = func(X, y)
        elif func is PCA_sort:
            ev, evr, ranks = PCA_sort(X)
        elif func is dropcol_importance:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            ranks = dropcol_importance(model, X_train, y_train, X_val, y_val)
        elif func is permutation_importance:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
            ranks = permutation_importance(model, X_val, y_val)
        
        acc = []
        for i in range(1, k+1):
            model_ = clone(model)
            features = [a[0] for a in ranks[:i]]
            X_ = X[features].copy()
            score = cross_val_score(model_, X_, y, cv=folds, scoring=metric)
            acc.append(np.mean(score))
        all_acc.append(np.array(acc))
        
    spear, pca, dropcol, permute = all_acc[0], all_acc[1], all_acc[2], all_acc[3]
    plt.plot(list(range(1, k+1)), spear, 'o-', c='blue', label='Spearman')
    plt.plot(list(range(1, k+1)), pca, '^-', c='red', label='PCA')
    plt.plot(list(range(1, k+1)), dropcol, 's-', c='green', label='DropColumn')
    plt.plot(list(range(1, k+1)), permute, 'd-', c='black', label='Permutation')
    plt.xlabel('Top k most important features')
    plt.ylabel('5-Fold CV Accuracy')
    plt.title(f'{model}')
    plt.legend()