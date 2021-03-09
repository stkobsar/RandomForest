from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

def get_dataframe(cancer):
    df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                      columns= np.append(cancer['feature_names'], ['target']))
    return df_cancer

def get_info_dataframe(df):
    with open('dataset_info.txt','w') as out:
      df.info(buf = out)

if __name__ == "__main__":

    working_df = get_dataframe(load_breast_cancer())
    print(working_df.head())

    get_info_dataframe(working_df)


