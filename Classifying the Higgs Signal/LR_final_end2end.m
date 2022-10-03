%% Machine Learning Coursework
% Logistic Regression Model

clear all; clc;

%% Importing data

load training_data.mat;
load test_data.mat;

%% Separating dependent and independent variables
X_train = train(:,1:9);
y_train = train(:,10);

X_test = test(:,1:9);
y_test = test(:,10);

%% Building logistic regression model 

%% Cross Validation to tune hyperparameters

% Ensures model reproducability
rng("default")

% Performing 5-fold cross-validation to optimise hyperparameters
[cv_error, optimal_func] = LR_kfoldcv(train, 5);

%% Training

% Building logistic regression model with optimised hyperparameters
model = fitclinear(train,'Label', 'Learner', 'logistic','lambda',0,"Solver","bfgs");

% Saving model
% save("LR_trained_model.mat", "model");

% Calculating training error
train_error = loss(model, train, "Label");

%% Testing

% Calculating test error
test_error = loss(model, test, "Label");

%% Model Evaluation

% Making predictions on test set
[~, score] = predict(model, X_test);

% Setting threshold 
threshold = 0.4;
y_pred = double(score(:,2) > threshold);

% Making variable compatible with perfcurve and custom function
y_test = table2array(y_test);

%% Plot ROC curve

% Calculate AUC
[X Y T AUC] = perfcurve(y_test, score(:,2), 1);

% Plot ROC curve
figure;
plot(X,Y);
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title("ROC Curve for Logistic Regression");

%% Calculating Evaluation Metrics and Plotting Confusion Matrix

% Using custom function - model_evaluation
[prec, rec, f_score] = model_evaluation(y_test, y_pred);