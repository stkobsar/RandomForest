from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

def get_dataframe(cancer):
    """
    Description: Get breast cancer data from sklearn datasets
    :param cancer: sklearn dataset
    :return: pandas dataframe
    """
    df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                      columns= np.append(cancer['feature_names'], ['target']))
    return df_cancer

def get_info_dataframe(df):
    """
    Returns information of the datafram in a txt file
    :param df: breast cancer dataframe
    :return: dataset_info.txt file
    """
    with open('dataset_info.txt','w') as out:
      df.info(buf = out)



