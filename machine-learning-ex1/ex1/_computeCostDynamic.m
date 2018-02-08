function J = _computeCostDynamic(X, y, theta, m)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


%J = (2*m)^-1 * sum((X*theta - y) .^2); % use element mutiplier (perioid, .)
J = (2*m)^-1 * (X*theta - y)' * (X*theta - y);  % an alternative implementation


% =========================================================================

end
