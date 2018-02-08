function J = costFunction(X,Y,theta)

% X is the "design matrix", containing the training examples
% Y is the class labels
% theta are the coefficient / parameters for the hypothesis (prediction) function

m = size(X,1);              % number of training examples
predictions = X * theta;    % prediction of hypothesis on all m training examples
sqErr = (predictions - Y).^2; % squared errors - note the period to perform element-wise squaring

J = 1 / (2 * m) * sum(sqErr); % calculate cost function