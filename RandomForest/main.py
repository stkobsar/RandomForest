from sklearn.datasets import load_breast_cancer
import RandomForest.get_data as gd
import RandomForest.EDA as EDA

def main():
    working_df = gd.get_dataframe(load_breast_cancer())
    gd.get_info_dataframe(working_df)


    EDA.get_histograms(working_df)
    EDA.get_qqplot(working_df)
    EDA.correlation_matrix(working_df)



if __name__ == "__main__":
    main()