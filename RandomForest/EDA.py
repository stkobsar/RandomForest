import os
import matplotlib.pyplot as plt
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__)) # Figures out the absolute path for you in case your working directory moves around.
hist = "histograms"
my_path = os.path.join(path, hist)

def get_histograms(df):

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            df.hist(column=col)
            plt.savefig(os.path.join(my_path, f"{col}.png"))
        except ValueError:
            print('This column can not be represented as an histogram')

