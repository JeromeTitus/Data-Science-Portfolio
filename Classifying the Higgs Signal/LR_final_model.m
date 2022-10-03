%% Machine Learning Coursework
% Logistic Regression - Testing

% NOTE: Workspace may get flooded due to imported scores for comparison.
% The evaluation metrics relevant to the logistic regression model are given
% by: AUC_lr, precision, recall, f_score and test_error

clear all; clc;

%% Loading data and model

% Loading test data
load test_data.mat;

% Loading trained model
load LR_trained_model.mat;

%% Testing model

% Calculating test error
test_error = loss(model, test, "Label");

% Making predictions on test set (probabilities)
[~, score_lr] = predict(model, test(:,1:9));

% Obtaining target variable from test set
y_test = table2array(test(:,10));

%% Calculating AUC and Plotting ROC

% Calculate AUC 
[Xlr Ylr Tlr AUC_lr] = perfcurve(y_test, score_lr(:,2), 1);

% Loading scores from decision tree to faciliate comparison
load DT_score.mat

% Decision Tree values 
[Xdt, Ydt, Tdt, AUCdt] = perfcurve(y_test, score(:,2), 1);

% Random classifier values
X = 0:0.1:1;
Y = X;

% Plot ROC curve
figure;
plot(Xlr,Ylr, "LineWidth", 3);
hold on
plot(Xdt, Ydt, "LineWidth", 3);
plot(X,Y, "black", "LineWidth",3);
xlabel("False Positive Rate");
ylabel("True Positive Rate");
legend("Logistic Regression", "Decision Tree", "Random Classifier","Location","Best");
title("ROC Curves");


%% Calculating Evaluation Metrics and Plotting Confusion Matrix

% Obtaining labels according to threshold
% Note: Initial recall and F1_scores were poor. This is generally due to
% poor threshold values. As a result, the threshold was reduced to 0.4.

threshold = 0.4;

y_pred = double(score_lr(:,2) > threshold);

[precision, recall, f_score] = model_evaluation(y_test, y_pred);