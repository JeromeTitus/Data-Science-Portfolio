Machine Learning: Coursework


This file will contain instructions on how to run models, including library or directory dependencies and required software versions.



###################################### MODEL TESTING #######################################

To test decision tree model, run DT_final_model.m

To test logistic regresion model, run LR_final_model.m

############################################################################################



##################################### CODE BREAKDOWN #######################################

DT_final_model.m - Testing script for decision tree
DT_final_end2end.m - Training and testing script for decision tree including hyperparameter optimization
DT_kfoldcv.m - Custom function script to perform k-fold cross-validation for decision tree
DT_trained_model.mat - Saved decision tree model (trained model)
DT_score.mat - Saved decision tree score (for comparison with logistic regression)


LR_final_model.m - Testing script for decision tree
LR_final_end2end.m - Training and testing script for logistic regression including hyperparameter optimization
LR_kfoldcv.m - Custom function script to perform k-fold cross-validation for logistic regression
LR_trained_model.mat = Saved logistic regression model (trained model)

model_evaluation.m - Custom function to calculate evaluation metrics

training_data.mat - Training data
test_data.mat - Test data

ML_Coursework_Preprocessing.html - Converted Jupyter Notebook containing data preprocessing and exploratory data analysis


#############################################################################################



################################### ADDITIONAL FILES ########################################

data_cleaned.csv - Cleaned dataset 
training.csv - Original dataset obtained from Kaggle


##############################################################################################





##################################### SOFTWARE DEPENDENCIES ################################

'MATLAB' Version 9.11 (R2021b)
'Statistics and Machine Learning Toolbox' Version 12.2 (R2021b)


Python

pandas Version 1.2.4
numpy Version 1.20.1
matplotlib Version 3.3.4
seaborn Version 0.11.1


############################################################################################

