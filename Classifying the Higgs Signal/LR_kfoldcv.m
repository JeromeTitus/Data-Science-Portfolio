%% Function to perform k-fold cross validation for logistic regression

function [crossval_err, opt_func] = LR_kfoldcv(data, K)

    % There will only be one hyperparameter tuned - objective function
    % minimization technique
    
    % Best function (placeholder for now)
    opt_func = 0;
    
    % Minimum error (placeholder for now)
    crossval_err = 100;
    
    % List of functions
    opt = ["sgd"; "asgd"; "bfgs"; "lbfgs"];
    n = numel(opt);
    
    % Running k-fold cross-validation
    for i = 1:n,
    
        min_func = opt(i);
    
        cvmodel = fitclinear(data, 'Label','Lambda',0,'Learner','logistic','Solver',min_func,'KFold',K);
    
        cv_loss = kfoldLoss(cvmodel);
    
        if cv_loss < crossval_err,
    
            crossval_err = cv_loss;
            opt_func = min_func;
    
        end
    end
end