%% Machine Learning Coursework
% Decision Tree - Testing 

clear all; clc;

%% Loading data and model

% Loading test data
load test_data.mat;

% Loading trained model
load DT_trained_model.mat;

%% Testing model

% Calculating test error
test_error = loss(model, test, "Label");

% Making predictions on test set
[~, score] = predict(model, test(:,1:9));

% Save score for comparison of ROC curves with logistic regression
% save("DT_score.mat", "score");

% Obtaining predicted labels (not probabilities as above)
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
title("ROC Curve");


%% Calculating Evaluation Metrics and Plotting Confusion Matrix

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