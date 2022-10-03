%% Creating a function to calculate evaluation metrics and plot confusion matrix

% Given imbalance of target variable, these metrics will be optimal for
% model evaluation 
function [prec, rec, f_score] = model_evaluation(ytest, yfit)

    % Plotting confusion matrix
    figure;
    conf_matrix = confusionchart(ytest, yfit);

    % Obtaining values for confusion matrix
    c = confusionmat(ytest, yfit);

    TP = c(2,2);
    TN = c(1,1);
    FP = c(1,2);
    FN = c(2,1);

    % Calculating precision
    prec = TP / (TP + FP);

    % Calculating recall
    rec = TP / (TP + FN);

    % Calculating F-score
    f_score = (2 * prec * rec) / (prec + rec);
end