import json
import time
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# plt.rcParams['figure.figsize'] = (10, 10)

# index represents label: 0 BAD, 1 GOOD, 2 ANOM
palette = ["red", "green", "orange"]


def prepare_df(df, X, y, bin_cols=None, ignore_y=False):

    if df is None and (X is None or y is None):
        print("Should provide df or (X and y)")

    if X is None or y is None:

        if bin_cols is None:
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


def do_pca(df=None, X=None, y=None, bin_cols=None, include_elbow=False, ignore_y=False, title="", show=False, save_path=False):
    X, y = prepare_df(df, X, y, bin_cols, ignore_y)

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
        plt.ylabel('Variance (%)')  # for each component
        plt.title('Explained Variance ratio')
        plt.show()

    colors = [palette[y_] for y_ in y]

    plt.figure(figsize=(20, 20))
    plt.title(title)
    plt.scatter(pcomp[:, 0], pcomp[:, 1], color=colors, alpha=.1, label=y)
    
    plt.axis('off')
    if save_path:
        # plt.savefig(Path(save_path).with_suffix(".svg"), format='svg')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()


def do_tsne(df=None, X=None, y=None, bin_cols=None, ignore_y=False, metric=None, show=False, save_path=None):

    X, y = prepare_df(df, X, y, bin_cols, ignore_y)

    if metric is not None:
        tsne = TSNE(n_components=2, random_state=42, metric=metric)
    else:
        tsne = TSNE(n_components=2, random_state=42)

    pcomp = tsne.fit_transform(X)

    colors = [palette[y_] for y_ in y]

    plt.figure(figsize=(20, 20))
    plt.scatter(pcomp[:, 0], pcomp[:, 1], color=colors, alpha=.1, label=y)
    
    plt.axis('off')
    if save_path:
        # plt.savefig(Path(save_path).with_suffix(".svg"), format='svg')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()


def df_plot(df, bin_cols=None, title="", show=False, save_path=None, ymax=0.05):
    plt.figure(figsize=(20, 20))
    plt.title(title)

    for _, row in df.iterrows():

        if bin_cols is None:
            data = [row[col]/row["entries"] for col in row.keys() if 'bin_' in col]
        else:
            data = [row[col]/row["entries"] for col in bin_cols]

        data = data[1:-1]

        plt.plot(range(len(data)), data, color=palette[row['y']], alpha=.1)
        
    if ymax is not None:
        plt.ylim(ymax=ymax) # do not move this line

    plt.axis('off')
    if save_path:
        # plt.savefig(Path(save_path).with_suffix(".svg"), format='svg')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()


def df_plot2(df_left, df_right, bin_cols=None, title1="", title2="", show=False, save_path=None, ymax=0.05):
    plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(121)
    ax1.set_title(title1)
    
    for _, row in df_left.iterrows():

        if bin_cols is None:
            data = [row[col]/row["entries"] for col in row.keys() if 'bin_' in col]
        else:
            data = [row[col]/row["entries"] for col in bin_cols]

        data = data[1:-1]

        ax1.plot(range(len(data)), data, color=palette[row['y']], alpha=.1)

    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    ax2.set_title(title2)
    
    for _, row in df_right.iterrows():

        if bin_cols is None:
            data = [row[col]/row["entries"] for col in row.keys() if 'bin_' in col]
        else:
            data = [row[col]/row["entries"] for col in bin_cols]

        data = data[1:-1]

        ax2.plot(range(len(data)), data, color=palette[row['y']], alpha=.1)
    
    if ymax is not None:
        plt.ylim(ymax=ymax) # do not move this line
    
    if save_path:
        # plt.savefig(Path(save_path).with_suffix(".svg"), format='svg')
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()


def do_gif(save_dir, directory="pca"):
    cmd = f"cd {save_dir} ; convert -loop 0 `ls {directory} -v | grep '^[0-9]'` training.gif"
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()


if __name__ == "__main__":

    pass
