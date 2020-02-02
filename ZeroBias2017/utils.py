""" utils.py contains classes and functions to help working with UL datasets """


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GoldenJSON():
    """ GoldenJSON - contains list of GOOD only lumisections"""

    def __init__(self, filename):
        """
            filename(str): fullpath to golden json file
        """

        self.golden_json = None

        with open(filename, "r") as f:
            self.golden_json = json.load(f)

    def is_good(self, row, run_column_name="run", lumisection_column_name="lumi"):
        """ Returns 1/0 if lumisection is GOOD/BAD respectively

            Parameters
            ----------
                row : pd.Series
                run_column_name : str
                lumisection_column_name : str
        """

        run = str(row[run_column_name])
        ls = row[lumisection_column_name]

        if run not in self.golden_json:
            return 0

        # Check if lumisection number is between list of ranges
        for ls_range in self.golden_json[run]:
            if ls_range[0] <= ls <= ls_range[1]:
                return 1

        return 0


class BadJSON():
    """ BadJSON - similar to golden json, but contains list of BAD only lumisections """

    def __init__(self, filename):
        """
            filename(str): fullpath to bad json file
        """

        self.bad_json = None

        with open(filename, "r") as f:
            self.bad_json = json.load(f)

    def is_good(self, row, run_column_name="run", lumisection_column_name="lumi"):
        """ Returns 1/0 if lumisection is GOOD/BAD respectively

            Parameters
            ----------
                row : pd.Series
                run_column_name : str
                lumisection_column_name : str
        """

        run = str(row[run_column_name])
        ls = row[lumisection_column_name]

        if run not in self.bad_json:
            return 1

        # Check if lumisection number is between list of ranges
        for ls_range in self.bad_json[run]:
            if ls_range[0] <= ls <= ls_range[1]:
                return 0

        return 1


def transform_histo_to_columns(df, histo_column_name="histo", col_prefix="bin_"):
    """ This function expands list as string to a N of columns for each value in a list

        Parameters
        ----------
            df : pd.DataFrame
            histo_column_name : str
                column name which holds list as a string
            col_prefix : str
                prefix of new column names

        Returns
        -------
        new pandas dataframe

    """

    def convert(histo):
        histo = histo.replace("[", "").replace("]", "").replace(" ", "")
        l = [int(i) for i in histo.split(",")]
        return pd.Series(l)

    # Expand histo into new dataframe
    histo = df[histo_column_name].apply(convert)

    # Rename new columns
    histo = histo.rename(columns=lambda x: col_prefix + str(x))

    df_new = pd.concat([df.drop(histo_column_name, axis=1), histo], axis=1)

    return df_new


def view_histo_from_raw(row, show=True, save_path=None):
    """ View or Save histogram from row of original (not massaged) dataset"""

    hname = row["hname"]
    run_number = row["run"]
    ls_number = row["lumi"]
    is_good = row["good"]
    y = row["ybins"]
    x = row["xbins"]

    histostr = row["histo"].replace("[", "").replace("]", "").replace(" ", "")
    data = np.fromstring(histostr, sep=',')


    plt.figure(figsize=(10, 5))
    plt.title("%s Run: %s LS: %s Label: %d" % (hname, run_number, ls_number, is_good))

    if row["metype"] == 3:
        # 1D
        plt.plot(range(len(data)), data, drawstyle='steps-pre', label=hname)

    elif row["metype"] == 6:
        # 2D
        data = data.reshape(y+2, x+2)

        # Remove border
        data = data[1:-1, 1:-1]

        plt.colorbar(plt.pcolor(data, cmap="rainbow"))

    plt.legend()

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path)
    plt.close()
