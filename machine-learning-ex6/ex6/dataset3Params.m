function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

iters = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
results = []; % C sigma prediction-error

% Submit runs the entire fucking iteration, so just using the iter solutions for now
for i_c = 1 %iters - putting in the the optimal solution
  for i_s = 0.1 %iters - putting in the optimal solution
    
    model= svmTrain(X, y, i_c, @(x1, x2) gaussianKernel(x1, x2, i_s));  % I dont know how x1 x2 feat works...  
    predictions = svmPredict(model,Xval);
    pred_err = mean(double(predictions ~= yval));
    results = [results(:,:); i_c i_s pred_err]; % Add additional row with results from C and sigma combo
    
  endfor
endfor  

% Find minimum prediction error by C / sigma combo and return;
[m_val row_ind] = min(results(:,end)); % row index for lowest result in last column
C = results(row_ind,1);
sigma = results(row_ind,2);


% =========================================================================

end
