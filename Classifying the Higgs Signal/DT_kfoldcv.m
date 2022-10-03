%% Creating a function to perform cross validation for a decision tree

function [min_error, leaves, splits] = DT_kfoldcv(data, K)

    % Hyperparameters with lowest error (placeholder values for now)
    leaves = 0;
    splits = 0;

    % Minimum error (placholder value for now)
    min_error = 100;

    % Counter - tracking model run
    counter = 0;

    % Performing k-fold cross validation
    % Steps used to optimize running time
    for l = 1:10:100,

        for s = 1:10:100,

            
            tree = fitctree(data, "Label" ,'MinLeafSize', l,'MaxNumSplits',s,'kFold',K);
            loss = kfoldLoss(tree);
           
            % Saving hyperparameters with lowest error
            if loss < min_error,
    
                min_error = loss;
                leaves = l;
                splits = s;
    
            end
            

            counter = counter + 1


        end
    end       
end
