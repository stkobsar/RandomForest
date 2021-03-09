from sklearn.datasets import load_breast_cancer
import RandomForest.get_data as gd

def main():
    working_df = gd.get_dataframe(load_breast_cancer())
    print(working_df.head())


if __name__ == "__main__":
    main()