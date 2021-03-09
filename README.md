# RandomForest

## Random Forest algorithm use case

Random forests is a supervised learning algorithm. It can be used both for classification 
and regression. It is also the most flexible and easy to use algorithm. A forest is 
comprised of trees. It is said that the more trees it has, the more robust a forest is. 
Random forests creates decision trees on randomly selected data samples, gets prediction
from each tree and selects the best solution by means of voting. It also provides a 
pretty good indicator of the feature importance.

Random forests has a variety of applications, such as recommendation engines, 
image classification and feature selection. It can be used to classify loyal loan 
applicants, identify fraudulent activity and predict diseases. It lies at the base of 
the Boruta algorithm, which selects important features in a dataset.

The individual decision trees are generated using an attribute selection indicator such as 
information gain, gain ratio, and Gini index of impurity for each attribute. Each tree depends on an 
independent random sample. In a classification problem, each tree votes and the most 
popular class is chosen as the final result. In the case of regression, the average of 
all the tree outputs is considered as the final result. It is simpler and more powerful 
compared to the other non-linear classification algorithms.

In this case, we are going to apply Random Forest algorithm to the Breast Cancer dataset 
available in the sklearn datasets library to classify if it benign or malign tumor. 