import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# plt.rcParams['figure.figsize'] = (10, 10)

# index represents label: 0 BAD, 1 GOOD, 2 ANOM
palette=["red", "green", "orange"] 

def prepare_df(df, X, y, ignore_y=False):

    if df is None and (X is None or y is None):
        print("Should provide df or (X and y)")
    
    if X is None or y is None:
        # Filter list of columns which will be used for training
        bin_cols = [col for col in df.columns if 'bin_' in col]

        # remove first and last values as those are over/under flows
        bin_cols = bin_cols[1:-1]

        X = df.filter(bin_cols, axis=1).copy().div(df.entries, axis=0)
        
        if ignore_y:
            y = [2 for _ in range(len(df))]
        else:
            y = df["y"]

    return X, y

def do_pca(df=None, X=None, y=None, include_elbow=False, ignore_y=False, title="", show=False, save_path=False):
    
    X, y = prepare_df(df, X, y, ignore_y)
        
    n_components = len(y)
    if n_components > 4:
        n_components = 4
    
    pca = PCA(n_components=n_components, random_state=42)
    pcomp = pca.fit_transform(X)
    
    if include_elbow:
        print("Explained variance ratio", pca.explained_variance_ratio_)
        plt.figure()
        plt.grid()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Variance (%)') #for each component
        plt.title('Explained Variance ratio')
        plt.show()
    
    colors = [palette[y_] for y_ in y]
    
    plt.figure()
    plt.title(title)
    plt.scatter(pcomp[:,0], pcomp[:,1], color=colors, alpha=.1, label=y)
    
    if save_path:
        # plt.savefig(Path(save_path).with_suffix(".svg"), format='svg')
        plt.savefig(save_path)

    if show:
        plt.show() 
    
    plt.close() 

def do_tsne(df=None, X=None, y=None, ignore_y=False, metric=None, show=False, save_path=None):
    
    X, y = prepare_df(df, X, y, ignore_y)
    
    if metric is not None:
        tsne = TSNE(n_components=2, random_state=42, metric=metric)
    else:
        tsne = TSNE(n_components=2, random_state=42)
        
    pcomp = tsne.fit_transform(X)
    
    colors = [palette[y_] for y_ in y]
    
    plt.figure(figsize=(20, 20))
    plt.scatter(pcomp[:,0], pcomp[:,1], color=colors, alpha=.1, label=y)

    if save_path:
        # plt.savefig(Path(save_path).with_suffix(".svg"), format='svg')
        plt.savefig(save_path)

    if show:
        plt.show() 
    
    plt.close()
