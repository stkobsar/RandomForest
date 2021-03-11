from sklearn.datasets import load_breast_cancer
import RandomForest.get_data as gd
import RandomForest.EDA as EDA
import RandomForest.rf_model as rf

def main():

    #### Get Data ####

    working_df = gd.get_dataframe(load_breast_cancer())
    gd.get_info_dataframe(working_df)
    gd.get_data_statistics(working_df)

    #### Exploratory Data Analysis ####

    EDA.get_histograms(working_df)
    EDA.get_qqplot(working_df)
    EDA.correlation_matrix(working_df)

    #### Training and testing the model ####

    X_train_scaled, X_test_scaled, y_train, y_test = rf.data_preparing(working_df)
    confusion_matrix, cross_val_score, model_accuracy = rf.RandomForestModel(X_train=X_train_scaled, X_test=X_test_scaled, y_train=y_train, y_test=y_test)

    print("\nRESULT OF THE CLASSIFICATION OF MALIGN AND BENIGN BREAAST TUMOR USING RANDOM FOREST ALGORITHM: \n")
    print(f"The confusion matrix of the prediction is: \n {confusion_matrix} \n")
    print(f"The 10-fold cross validation score is {cross_val_score*100:.{2}f} % \n")
    print(f"The model accuracy is {model_accuracy*100:.{2}f} % \n")

if __name__ == "__main__":

    main()
