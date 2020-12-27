try:
    import waitGPU
    waitGPU.wait(ngpu=1)
except:
    print("Failed to import waitGPU")

from utils import do_tsne, do_pca, do_gif, df_plot
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
import os
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class BlackBox:
    def __init__(self, data_dir, save_dir="trainings", threshold=0.9):
        self.data_dir = data_dir

        _date_time = datetime.now() .strftime("%Y-%m-%dT%H-%M-%S")
        current_dir = f"train-{Path(data_dir).name}-t-{threshold}-{_date_time}"

        self.save_dir = Path(save_dir).joinpath(current_dir)
        self.save_dir.mkdir(parents=True)

        print(f"Save dir: {self.save_dir}")

        self.df_train = None
        self.df_test = None
        self.bin_cols = []

        self.df_anomalies = None
        self.threshold = threshold

        self.training_performance = []
        self.model = None

    def load_df(self, nbins=100, step=1):

        self.df_train = pd.read_csv(
            Path(self.data_dir, 'train.csv'))   # Hand picked data
        self.df_test = pd.read_csv(
            Path(self.data_dir, 'test.csv'))  # Data to be classified

        print(
            f"Train set {self.df_train.shape} | Test set {self.df_test.shape}")

        # Filter list of columns which will be used for training
        self.bin_cols = [col for col in self.df_train.columns if 'bin_' in col]

        # Remove first and last values as those are over/under flows
        self.bin_cols = self.bin_cols[1:-1]

        if nbins > len(self.bin_cols):
            nbins = len(self.bin_cols)

        self.bin_cols = [f"bin_{i}" for i in range(1, nbins, step)]

    def create_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(units=50, activation="relu",
                               input_shape=(len(self.bin_cols),)))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units=25, activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units=1, activation="sigmoid"))
        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def train_model(self, df):
        """ Trains new model with a given dataset
            Params:
                df (pandas.DataFrame) - dataset for training new model

            Returns:
                model (keras.Sequential) - trained model
        """
        try:
            # Normalization, divide every bin value by total entries
            X = df.filter(self.bin_cols, axis=1).copy().div(df.entries, axis=0)
            y = df["y"]

            # Stratified shuffle split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, stratify=y, test_size=0.25)

            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

            # Calculate class weights
            classes = np.unique(y)
            weights = class_weight.compute_class_weight('balanced', classes, y)
            cw = {int(cls): weight for cls, weight in zip(classes, weights)}

            # Train
            es = EarlyStopping(monitor='loss', mode='min',
                               patience=5, verbose=1)
            model = self.create_model()
            model.fit(X_train, y_train, verbose=0,
                      batch_size=150,
                      epochs=1000,
                      shuffle=True,
                      callbacks=[es])
                    #   class_weight=cw)

            # Predict
            y_pred_ = model.predict(X_test)
            y_pred = (y_pred_ > 0.5)

            self.training_performance.append({
                "acc": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "df_size": df.shape[0],
                "ngood": df[df["y"] == 1].shape[0],
                "nbad": df[df["y"] == 0].shape[0],
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            })
        except Exception as err:
            print("Failed train_model |", err)
            raise err

        return model

    def predict_run(self, df_run):
        """ Does prediction with given model and dataset

            df_run (pd.DataFrame) - dataset

            Returns:
                df_run (pd.DataFrame) - dataset with additional two columns: 
                                        y_pred - model prediciton [0..1]
                                        y with - label (1 good, 0 bad, 2 anomaly)

        """

        try:
            # Scale each row by dividing by number of total entries within a run
            X = df_run.filter(self.bin_cols, axis=1).div(
                df_run.entries, axis=0)
            X = np.asarray(X)

            y_pred = self.model.predict(X)

            # Predicted label by ANN
            df_run["y_pred"] = y_pred

            # Create new column y and set final label there

            # Predictions with higher probability than threshold are considered as GOOD
            filter_good = df_run['y_pred'] >= self.threshold
            # Predictions with lower probability than threshold are considered as BAD
            filter_bad = df_run['y_pred'] <= 1-self.threshold
            # Predictions between lower and higher thresholds are considered as ANOMALIES
            filter_anon = (
                1-self.threshold < df_run['y_pred']) & (df_run['y_pred'] < self.threshold)

            # Create new column y and set final label there
            df_run.loc[filter_good, 'y'] = 1
            df_run.loc[filter_bad, 'y'] = 0
            df_run.loc[filter_anon, 'y'] = 2
            df_run = df_run.astype({"y": "int32"})

        except Exception as err:
            print("Failed predict_run |", err)
            raise err

        return df_run

    def self_train(self, nruns=None):
        self.t1 = time.time()
        print("Training initial model", end=' | ')
        self.model = self.train_model(self.df_train)

        runs = self.df_test.run.unique()

        if nruns:
            runs = runs[:nruns]

        for train_index, run_number in enumerate(runs):
            print(f"{train_index}. Working with run {run_number}", end=' | ')

            try:
                # Dataset of a single run
                df_run = self.df_test[self.df_test["run"] == run_number].copy()

                if len(df_run) == 0:
                    print(f"Run {run_number} has no data in test dataset")
                    continue

                df_run = self.predict_run(df_run)

                # Take a subset of only anomalies (y=2)
                df_anomalies = df_run[df_run["y"] == 2].copy()
                if self.df_anomalies is None:
                    self.df_anomalies = df_anomalies
                else:
                    self.df_anomalies = pd.concat([self.df_anomalies, df_anomalies],
                                                  ignore_index=True, sort=False)

                # Take a subset of only good and bad predictions, but no anomalies
                df_confident = df_run[df_run["y"] != 2].copy()

                # Add new predictions to a training dataset
                self.df_train = pd.concat([self.df_train, df_confident],
                                          ignore_index=True, sort=False)

                pca_save_path = self.save_dir.joinpath(f"pca/{train_index}-{run_number}.jpg")
                dftrain_save_path = self.save_dir.joinpath(f"dftrain/{train_index}-{run_number}.jpg")

                do_pca(self.df_train,
                       title=f"{train_index}", save_path=pca_save_path)

                df_plot(self.df_train, title=f"{train_index}", save_path=dftrain_save_path)
                
                self.model = self.train_model(self.df_train)
            except Exception as err:
                print("Failed self_train |", err)

        self.t2 = time.time()

    def save(self):

        self.model.save(self.save_dir.joinpath('model.h5'))

        self.df_train.to_csv(self.save_dir.joinpath('df_train.csv'))
        self.df_anomalies.to_csv(self.save_dir.joinpath('df_anomalies.csv'))

        # Make a gif from pca images
        # do_gif(self.save_dir)

        df_ta = pd.concat([self.df_train, self.df_anomalies],
                          ignore_index=True, sort=False)

        # TSNE for classified dataset
        do_tsne(self.df_train, save_path=self.save_dir.joinpath(
            f"tsne_df_train.jpg"))

        # TSNE for classified dataset + anomalies
        do_tsne(df_ta, save_path=self.save_dir.joinpath(f"tsne_df_ta.jpg"))

        # Plot histograms of classified dataset
        df_plot(self.df_train, save_path=self.save_dir.joinpath(f"df_train.jpg"))

        # Plot histograms of anomalies
        df_plot(self.df_anomalies, save_path=self.save_dir.joinpath(
            f"df_anomalies.jpg"))

        # Plot histograms of classified dataset + anomalies
        df_plot(df_ta, save_path=self.save_dir.joinpath(f"df_ta.jpg"))

        self.t3 = time.time()

        self.training_performance = {
            "trainings": self.training_performance,
            "duration": {
                "train": int(self.t2 - self.t1),
                "visual": int(self.t3 - self.t2),
                "total": int(self.t3 - self.t1)
            }
        }

        with open(self.save_dir.joinpath('training_performance.json'), 'w') as fh:
            json.dump(self.training_performance, fh)
        
        print(self.training_performance["duration"])


if __name__ == "__main__":

    batch = True

    # base_dir = Path('/eos/home-m/mantydze/ZeroBias2018B/scripts')
    base_dir = Path('.')

    if not batch:
        box = BlackBox(data_dir=base_dir.joinpath("data"),
                       save_dir=base_dir.joinpath("trainings_test"),
                       threshold=0.9)
        box.load_df()
        box.self_train(nruns=2)
        box.save()

        exit()

    else:
        data_dirs = ["data_single"]
        thresholds = sorted([0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95], reverse=True)

        for data_dir in data_dirs:
            for threshold in thresholds:
                box = BlackBox(data_dir=base_dir.joinpath(data_dir),
                            save_dir=base_dir.joinpath("trainings_single"),
                            threshold=threshold)
                box.load_df()
                box.self_train()
                box.save()
                print()
