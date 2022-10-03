%% Machine Learning Coursework
% Decision Tree Model 

clear all; clc;

%% Importing data

data = readtable("data_cleaned.csv", "PreserveVariableNames", true);

%% Separating data into train and test sets

% Ensuring model reproducability
rng("default") 

% Number of observations in dataset
n = size(data,1);    

% Separating training and test datasets
c = cvpartition(n, 'HoldOut', 0.3); 

% Training data
train = data(training(c), :);

% Saving training data for use in logistic regression model and testing
% save("training_data.mat", "train");

% Test data
test = data(test(c),:);

% Saving test data for use in logistic regression model and testing
% save("test_data.mat", "test");

%% Building decision tree model

%% Cross validation to tune hyperparameters

% Optimizing hyperparameters with 5-fold cross-validation using custom
% function - DT_kfoldcv. Evaluation with 10-fold cross-validation yielded 
% very similar results. 5-fold selected to optimize model run-time.
[cv_error, min_l, min_s] = DT_kfoldcv(train, 5);

%% Training 

% Building decision tree with optimized hyperparameters
model = fitctree(train, "Label", "MinLeafSize",min_l,"MaxNumSplits",min_s);

% Saving model for testing
% save("DT_trained_model.mat", "model");

% Calculating training error
train_error = resubLoss(model);

%% Testing

% Calculating test error
test_error = loss(model, test, "Label");

% Making predictions on test set
[~, score] = predict(model, test(:,1:9));

label = predict(model, test(:,1:9));

% Obtaining target variable from test set
y_test = test(:,10);

% Making variable compatible with perfcurve and custom function
y_test = table2array(y_test);

%% Calculating AUC and Plotting ROC

% Calculate AUC
[X Y T AUC] = perfcurve(y_test, score(:,2), 1);

% Plot ROC curve
figure;
plot(X,Y);
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title("ROC Curve for Decision Tree");

%% Calculating Evaluation Metrics and Plotting Confusion Matrix

% Using custom function - model_evaluation
[precision, recall, f_score] = model_evaluation(y_test, label);

%% Determining Feature Importance

imp = predictorImportance(model);

% Plotting feature importance on bar chart
figure;
bar(imp);
title("Feature Importance");
ylabel("Estimates");
xlabel("Features");
h = gca;
h.XTickLabel = model.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';