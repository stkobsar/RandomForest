import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns

path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
hist = "histograms"
qqplot = "qqplots"
my_path_hist = os.path.join(path, hist)
my_path_qq = os.path.join(path, qqplot)

def get_histograms(df):
    """
    Get histograms of all variables of dataframe
    :param df: breast cancer dataframe
    :return: histograms in png stored in RandomForest/histograms
    """

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            df.hist(column=col)
            plt.savefig(os.path.join(my_path_hist, f"{col}.png"))
        except ValueError:
            print('This column can not be represented as an histogram')

def get_qqplot(df):
    """
    Get qqplot of all variables of dataframe
    :param df: breast cancer dataframe
    :return: histograms in png stored in RandomForest/histograms
    """

    for col in df.columns:
        try:
            fig, ax = plt.subplots()
            valores = pd.to_numeric(df[col])
            stats.probplot(valores, dist="norm", plot=ax)
            plt.savefig(os.path.join(my_path_qq, f"{col}_qqplots.png"))
        except ValueError:
            print('Not able to return qqplot')

def correlation_matrix(df):
    """
    Get the Perason's correlation coefficients represented in a heatmap
    :param df: breast cancer dataframe
    :return: heatmap_breastcancer.png
    """
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(30, 30))
    heatmap = sns.heatmap(corr, vmax=1, square=True, annot=True)

    fig = heatmap.get_figure()
    fig.savefig(os.path.join(path, "heatmap_breastcancer.png"))
